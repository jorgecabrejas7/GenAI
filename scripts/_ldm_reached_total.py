"""Exit 0 only when an LDM run has actually reached total_steps.

Used by scripts/ldm06_post.sh. "No training process" is not the same as
"training finished": a deliberate two-minute stop looks identical from the
process table, and acting on it once cost eleven hours of GPU generating
volumes against a mid-training checkpoint.
"""
import json
import pathlib
import sys

import yaml

d = pathlib.Path(sys.argv[1])
try:
    total = int(yaml.safe_load((d / "resolved_config.yaml").read_text())["training"]["total_steps"])
    last = 0
    with (d / "log.jsonl").open() as fh:
        for line in fh:
            if '"step"' in line:
                try:
                    last = max(last, int(json.loads(line).get("step", 0)))
                except Exception:
                    pass
    sys.exit(0 if last >= total - 1 else 1)
except Exception as exc:  # noqa: BLE001
    print(f"cannot tell whether {d} finished: {exc}", file=sys.stderr)
    sys.exit(1)
