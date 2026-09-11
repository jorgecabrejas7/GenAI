"""Prove the memorisation hardware path on two volumes, under a hard cap.

`poregen.eval_v4.memorisation` was written and unit tested without a GPU.  Four
of its steps have therefore never run against real data: loading the frozen
VAE, encoding a generated volume's patches, `decode_grey` on the r08 decoder,
and the two full-store passes over a 272 GB store.  Any one of them can fail in
a way no unit test reaches — a dtype, a device, a missing key, an allocation.

The full check rides inside `eval_v4 measure microstructure` and takes hours on
the GPU.  Finding a load error there means finding it after those hours, with
the queue behind it.  This runs the identical code path on ONE 192-cubed volume
and ONE 384-cubed multichunk volume first, so a failure costs minutes.

It is a smoke test and says so: the numbers it prints answer nothing about
memorisation, because two volumes against the full bank is not the assessment.
What it establishes is that the path runs, what it costs per volume, and what
the full pass will therefore cost.

THE CAP IS ENFORCED FROM OUTSIDE.  The search runs in a child process and this
parent polls its host RSS and its GPU memory.  A watchdog inside the child
would share the child's fate: an allocation that runs away takes the interpreter
with it, and nothing is left to report why.  On a breach the child is killed and
the breach is written out.  The GPU queue is never held by a run that has
already exceeded what it was given.

Usage:
    python scripts/analysis/memorisation_smoke.py --root runs/campaigns/12-eval-v4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: One 192-cubed volume and one 384-cubed multichunk volume: the two shapes
#: differ in the thing the rewrite changed — a 192-cubed volume is a single
#: chunk and fills only the `present` neighbour bucket, a 384-cubed one fills
#: the UNKNOWN bucket too.  A smoke test that only ran the first would leave
#: the new code path untested.
SMOKE_ASSESSMENTS = ("sampler", "multichunk")

POLL_SECONDS = 10.0


def _rss_gb(pid: int) -> float:
    """Host resident set size of `pid`, in GB. 0.0 once the process is gone."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _gpu_gb(pid: int) -> float:
    """GPU memory `pid` holds, in GB. 0.0 when it holds none or nvidia-smi is absent."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return 0.0
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0.0
    for row in out.splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return float(parts[1]) / 1024
            except ValueError:
                return 0.0
    return 0.0


def _child(args: argparse.Namespace) -> int:
    """Run the search and write the result. Executed in the child process."""
    from poregen.eval_v4.memorisation import memorisation

    kwargs = {"repo": str(REPO)}
    if not args.full:
        kwargs["assessments"] = SMOKE_ASSESSMENTS
        kwargs["max_cases_per_assessment"] = 1
    result = memorisation(args.root, **kwargs)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="the eval-v4 campaign root")
    ap.add_argument("--out", default=None, help="where the result JSON goes")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--max-gb", type=float, default=20.0,
                    help="cap on host RSS and on GPU memory, each")
    ap.add_argument("--full", action="store_true",
                    help="run every volume, not the two-volume smoke set")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    tag = "full" if args.full else "smoke"
    args.out = args.out or str(Path(args.root) / f"memorisation_{tag}.json")

    if args._child:
        return _child(args)

    report = Path(args.root) / f"memorisation_{tag}_run.json"
    Path(args.root).mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, __file__, "--root", args.root, "--out", args.out,
           "--max-minutes", str(args.max_minutes), "--max-gb", str(args.max_gb),
           "--_child"]
    if args.full:
        cmd.append("--full")

    t0 = time.monotonic()
    # A new process group, so a breach kills the search and anything it spawned
    # rather than leaving a worker holding the GPU the queue is waiting for.
    child = subprocess.Popen(cmd, cwd=str(REPO), start_new_session=True)
    peak_rss = peak_gpu = 0.0
    breach: str | None = None

    while child.poll() is None:
        time.sleep(POLL_SECONDS)
        elapsed_min = (time.monotonic() - t0) / 60
        peak_rss = max(peak_rss, _rss_gb(child.pid))
        peak_gpu = max(peak_gpu, _gpu_gb(child.pid))
        if peak_rss > args.max_gb:
            breach = f"host RSS {peak_rss:.1f} GB > {args.max_gb:.1f} GB"
        elif peak_gpu > args.max_gb:
            breach = f"GPU memory {peak_gpu:.1f} GB > {args.max_gb:.1f} GB"
        elif elapsed_min > args.max_minutes:
            breach = f"elapsed {elapsed_min:.1f} min > {args.max_minutes:.1f} min"
        if breach:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait()
            break

    elapsed_min = (time.monotonic() - t0) / 60
    out = {
        "mode": tag,
        "assessments": None if args.full else list(SMOKE_ASSESSMENTS),
        "cases_per_assessment": None if args.full else 1,
        "elapsed_minutes": round(elapsed_min, 2),
        "peak_host_rss_gb": round(peak_rss, 2),
        "peak_gpu_gb": round(peak_gpu, 2),
        "cap_minutes": args.max_minutes,
        "cap_gb": args.max_gb,
        "breach": breach,
        "returncode": child.returncode,
        "result": args.out if breach is None and child.returncode == 0 else None,
    }
    report.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    if breach:
        print(f"STOPPED: {breach}", file=sys.stderr)
        return 2
    return 0 if child.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
