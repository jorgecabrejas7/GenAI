#!/usr/bin/env python
"""docs/PAPER_RUNBOOK.md — one row per experiment the paper uses. (CPU, seconds.)

GENERATED, NOT WRITTEN. The dataset is about to be rebuilt as split_v4
and every experiment must be rerunnable on the new pipeline, so the runbook has
to be regenerable too: a hand-written table drifts from the run directories the
moment anything is retrained, and a drifted runbook is worse than none because
it looks authoritative.

Everything mechanical — config file, dataset root, batch, steps reached, wall
time, checkpoint — is read from each run's own ``resolved_config.yaml`` and
``metrics.jsonl``. Everything editorial — what a campaign is for, its vault
note, whether its command still exists — is the CURATED tables below, which are
the part a human owns.

Where a command no longer exists the row says so rather than guessing.

Usage
-----
    python scripts/analysis/build_paper_runbook.py [--out docs/PAPER_RUNBOOK.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from poregen.paths import DEFAULT_SPLIT, SPLIT_ENV, repo_root

REPO = repo_root()

#: The runs the paper reports, in the order it reports them. A run not listed
#: here is a trial, a crash or a superseded attempt; runs/ holds many.
PAPER_RUNS: list[tuple[str, str, str]] = [
    # (section, run directory glob, what it is)
    ("VAE — the compression sweep", "runs/vae/r08-run-0010-*", "rf-2, z=32, 2x reduction"),
    ("VAE — the compression sweep", "runs/vae/r08-run-0006-*", "rf-4, z=16, 4x"),
    ("VAE — the compression sweep", "runs/vae/r08-run-0004-*", "rf-8, z=8, 8x — THE PAPER'S VAE"),
    ("VAE — the compression sweep", "runs/vae/r08-run-0003-*", "base, z=4, 16x"),
    ("VAE — the compression sweep", "runs/vae/r08-run-0005-*", "rf-32, z=2, 32x"),
    ("VAE — the compression sweep", "runs/vae/r08-run-0012-*", "rf-64, z=1, 64x"),
    ("VAE — rejected variants", "runs/vae/r08-run-0007-*", "decoder fine-tune — NEGATIVE, rejected"),
    ("LDM — the paper's model", "runs/ldm/ldm06-run-0001-*", "ldm06 base, 130k"),
    ("LDM — the paper's model", "runs/ldm/ldm06-run-0002-*", "facedrop fine-tune, 15k — THE PAPER'S LDM"),
    ("LDM — controls", "runs/ldm/ldm06-run-0003-*", "facedrop CONTROL (drop_nb_face 0)"),
    ("LDM — predecessors", "runs/ldm/ldm05-run-0001-*", "ldm05, the pre-r08 rung"),
    ("LDM — predecessors", "runs/ldm/ldm04-run-0001-*", "ldm04"),
    ("Baselines", "runs/ldm/ldm25-run-0001-*", "phi-only LDM (Naiff et al. 2026 recipe)"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae03-run-0003-*", "flat SVD bottleneck"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae04-run-0001-*", "flat linear twin"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae-run-0001-*", "V0, flat linear, KL off"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae-run-0006-*", "B, flat FC 1024, rank 768"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae-run-0009-*", "A0, flat SVD, KL off"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae-run-0008-*", "conv k*=8, beta 1e-5"),
    ("VRRAE family (campaign 28)", "runs/vae/vrrae-run-0010-*", "conv k*=8, beta 1e-3"),
]

#: Runs with no experiment config, driven by their own script.
SCRIPT_RUNS: list[dict] = [
    dict(section="Baselines", what="SliceGAN (Kench & Cooper 2021)",
         command="python scripts/train_slicegan.py",
         runs="runs/campaigns/22-slicegan-baseline/train",
         note="60 000 steps, 11.1 h, 13.38 M parameters. Unconditional.",
         vault="E15"),
    dict(section="Baselines", what="3-D pixel-space DDPM",
         command=f"python scripts/train_ddpm3d.py --data-root data/{DEFAULT_SPLIT}",
         runs="runs/campaigns/23-ddpm3d-baseline/train",
         note="25.33 M parameters, 64-cubed, v-objective, zero-terminal-SNR, "
              "24 h cap; measured at step 56 006 (ema). Unconditional.",
         vault="E15 (SliceGAN); no note of its own"),
]

#: The campaigns the paper draws on. Sourced from runs/campaigns/INDEX.md,
#: audited 2026-09-27; the vault column is what that audit verified.
CAMPAIGNS: list[dict] = [
    dict(n="12", name="eval-v4", what="ldm06 at 130k across 11 assessments, 177 cases",
         cmd="eval_v4 generate <assessment> --model <ldm06 run-0001> --ckpt 130000 --out runs/campaigns/12-eval-v4",
         vault="E8", caveat="README is still the pre-run stub — the campaign has no written conclusion."),
    dict(n="13", name="label-uncertainty", what="how much of a porosity number is label noise",
         cmd="python scripts/analysis/label_uncertainty.py", vault="E9", caveat=""),
    dict(n="14", name="downstream-utility", what="is the synthetic data USEFUL",
         cmd="python scripts/analysis/downstream_utility.py", vault="E14",
         caveat="Answer is NO; six arms, 18 runs."),
    dict(n="18", name="eval-v4-final", what="the same 12 assessments on the paper's weights and sampler",
         cmd="eval_v4 generate <assessment> --model <ldm06 run-0002> --ckpt latest --out runs/campaigns/18-eval-v4-final",
         vault="E10", caveat=""),
    dict(n="19", name="stress-geometry", what="which off-manifold shapes the envelope honours",
         cmd="eval_v4 generate stress_geometry --model <ldm06 run-0002> --ckpt latest --out runs/campaigns/19-stress-geometry",
         vault="E11", caveat="EXPLORATORY."),
    dict(n="20", name="ood-conditioning", what="conditioning outside the training range",
         cmd="eval_v4 generate ood_conditioning --model <ldm06 run-0002> --ckpt latest --out runs/campaigns/20-ood-conditioning",
         vault="E12", caveat="EXPLORATORY."),
    dict(n="22", name="slicegan-baseline", what="an unconditional 3-D GAN on our data, our eval",
         cmd="eval_v4 measure slicegan --root runs/campaigns/22-slicegan-baseline", vault="E15",
         caveat="Downstream arms still owed."),
    dict(n="23", name="ddpm3d-baseline", what="is the LATENT what makes ldm06 work",
         cmd="eval_v4 measure ddpm3d --root runs/campaigns/23-ddpm3d-baseline", vault="none — owed",
         caveat="On FID at its nearest level it BEATS ldm06 (3.63 against 5.91)."),
    dict(n="24", name="ablation", what="which of the seven conditioning inputs the model uses",
         cmd="eval_v4 generate ablation --model <ldm06 run-0002> --ckpt latest --out runs/campaigns/24-ablation",
         vault="E17", caveat=""),
    dict(n="25", name="ldm-phi-only", what="what the extra conditioning buys over the published recipe",
         cmd="eval_v4 generate <assessment> --model <ldm25 run-0001> --ckpt 40000 --out runs/campaigns/25-ldm-phi-only",
         vault="E16", caveat="Read WITH campaign 27; unmatched microstructure flatters the baseline."),
    dict(n="26", name="field-validation", what="does the coherent field deliver its correlation length",
         cmd="python scripts/analysis/porosity_field_validation.py", vault="none — owed",
         caveat="It delivered TWICE the requested length; sigma was L/stride, now L/(2*stride)."),
    dict(n="27", name="budget-matched-40k", what="conditioning or training budget?",
         cmd="eval_v4 generate <assessment> --model <ldm06 run-0001> --ckpt 40000 --out runs/campaigns/27-budget-matched-40k",
         vault="E16, section 'Budget-matched row'", caveat=""),
    dict(n="28", name="vrrae-family", what="the VRRAE bottleneck against the spatial baseline",
         cmd="python scripts/analysis/vrrae_family_table.py --out runs/campaigns/28-vrrae-family",
         vault="E18", caveat="Read texture and sharpness, not L1: five of eight rungs return flat grey."),
    dict(n="29", name="facedrop-control", what="did per-face dropout clear the band, or would any fine-tune?",
         cmd="eval_v4 generate multichunk --model <ldm06 run-0003> --ckpt latest --out runs/campaigns/29-facedrop-control",
         vault="E8, section 'Matched control'", caveat=""),
    dict(n="10/12", name="real-floor", what="what the request-free metrics score on REAL volumes",
         cmd="eval_v4 generate real-floor --out <campaign>", vault="E8",
         caveat="Run FIRST in any campaign: every ratio is against it."),
]


def _ckpt_name(d: Path) -> str:
    """Which checkpoint a rerun would load, and where it lives.

    VAE runs write best.ckpt beside the config; LDM runs write into
    checkpoints/. Looking only at the top level reported every LDM run in the
    paper as having no checkpoint at all, which was this file's first output.
    """
    for sub in ("", "checkpoints/"):
        for name in ("best.ckpt", "latest.ckpt"):
            if (d / sub / name).exists():
                return f"`{sub}{name}`"
    return "— none"


def run_facts(pattern: str) -> dict | None:
    dirs = sorted(REPO.glob(pattern))
    if not dirs:
        return None
    d = dirs[-1]
    cfg_p = d / "resolved_config.yaml"
    if not cfg_p.exists():
        return None
    cfg = yaml.safe_load(cfg_p.read_text())
    exp, data, tr = cfg.get("experiment", {}), cfg.get("data", {}), cfg.get("training", {})
    last, seen = None, 0
    # A RESUMED RUN RESTARTS ITS elapsed CLOCK. Each segment's elapsed counts
    # from that segment's launch, so the last row holds only the last segment:
    # ldm06 run-0001 read 16.1 h from it, for four segments that sum to 77.7 h.
    # A drop in elapsed marks a new segment; the wall time is the segment sum.
    seg_done, seg_max, segments = 0.0, 0.0, 1
    m = d / "metrics.jsonl"
    if m.exists():
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            try:
                last = json.loads(line); seen += 1
            except json.JSONDecodeError:
                continue
            e = last.get("elapsed")
            if e is None:
                continue
            if e < seg_max - 1.0:
                seg_done, segments = seg_done + seg_max, segments + 1
            seg_max = e
    eid = f"{exp.get('name')}/{exp.get('variant')}"
    cfg_file = f"configs/experiments/{eid}.yaml"
    return dict(
        run=d.name, eid=eid,
        config=cfg_file if (REPO / cfg_file).exists() else f"{cfg_file} (MISSING)",
        root=data.get("dataset_root") or data.get("latents_root") or "—",
        batch=data.get("batch_size") or cfg.get("data", {}).get("batch_size") or "—",
        planned=tr.get("total_steps") or "—",
        reached=(last or {}).get("step", "—"),
        hours=round((seg_done + seg_max) / 3600.0, 1),
        segments=segments,
        # LDM runs keep theirs in checkpoints/; VAE runs at the top level.
        ckpt=_ckpt_name(d),
        seed=tr.get("seed", "not recorded"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "PAPER_RUNBOOK.md")
    a = ap.parse_args()

    L: list[str] = []
    W = L.append
    W("# Paper runbook\n")
    W("**Generated by `scripts/analysis/build_paper_runbook.py`. Do not edit by hand** —")
    W("regenerate it. Every mechanical column is read from the run's own")
    W("`resolved_config.yaml` and `metrics.jsonl`; the editorial columns are curated")
    W("tables inside that script.\n")
    W(f"Every row below was measured on `{DEFAULT_SPLIT}`. To rerun the whole chain on a")
    W("new build, set the switch once — nothing else needs editing:\n")
    W("```bash")
    W(f"export {SPLIT_ENV}=split_v4")
    W("```\n")
    W(f"`{SPLIT_ENV}` overrides `data.dataset_root`, `data.latents_root` and")
    W("`data.split_version` in every config, and every script default reads")
    W("`poregen.paths`. See `tests/test_split_switch.py`.\n")
    W("**Wall time is the GB10's**, summed over the run's own metric rows, and it is")
    W("elapsed time including validation — not GPU time. A resumed run restarts its")
    W("clock, so its wall time is the sum of its segments and is marked; steps a")
    W("resume repeated are counted, because the card spent them. **Memory** is the whole")
    W("121 GB unified pool, shared by host and device: no second job may run beside")
    W("a training job. Batch sizes below are what fitted.\n")

    section = None
    for sec, pattern, what in PAPER_RUNS:
        if sec != section:
            section, _ = sec, W(f"\n## {sec}\n")
            W("| what | run | config | dataset root | batch | steps planned/reached | wall h | checkpoint |")
            W("|---|---|---|---|---|---|---|---|")
        f = run_facts(pattern)
        if f is None:
            W(f"| {what} | `{pattern}` | **NOT ON DISK** | — | — | — | — | — |")
            continue
        W(f"| {what} | `{f['run'][:44]}` | `{f['config']}` | `{f['root']}` | "
          f"{f['batch']} | {f['planned']} / {f['reached']} | {f['hours']}"
          + (f" ({f['segments']} segments, resumed)" if f["segments"] > 1 else "")
          + f" | {f['ckpt']} |")

    W("\n### Launch commands\n")
    W("```bash")
    W("# every VAE and LDM row above")
    W("python scripts/train_vae.py run <experiment>      # e.g. r08/reduction-factor-8")
    W("python scripts/train_ldm.py run <experiment>      # e.g. ldm06/facedrop")
    W("```")
    W("The experiment id is the `config` column with `configs/experiments/` and `.yaml`")
    W("stripped. Seeds are not a config key: the trainer seeds from the run name, so a")
    W("rerun of the same experiment is a different draw. **A rerun reproduces the")
    W("method, not the sample.**\n")

    W("\n## Baselines with their own trainer\n")
    W("| what | command | run directory | notes | vault |")
    W("|---|---|---|---|---|")
    for r in SCRIPT_RUNS:
        exists = (REPO / r["runs"]).exists()
        W(f"| {r['what']} | `{r['command']}` | `{r['runs']}`"
          f"{'' if exists else ' **(not on disk)**'} | {r['note']} | {r['vault']} |")

    W("\n## Campaigns\n")
    W("| # | campaign | question | command | vault | caveat |")
    W("|---|---|---|---|---|---|")
    for c in CAMPAIGNS:
        d = REPO / "runs" / "campaigns"
        here = sorted(d.glob(f"{c['n'].split('/')[0]}-*"))
        missing = "" if here else " **(not on disk)**"
        W(f"| {c['n']} | `{c['name']}`{missing} | {c['what']} | `{c['cmd']}` | "
          f"{c['vault']} | {c['caveat']} |")

    W("\n## What cannot be rerun as it stands\n")
    W("Regenerated with the rest of this file, so it cannot go stale silently.\n")
    gaps = []
    for sec, pattern, what in PAPER_RUNS:
        f = run_facts(pattern)
        if f is None:
            gaps.append(f"- **{what}** — no run directory matches `{pattern}`.")
        elif "MISSING" in f["config"]:
            gaps.append(f"- **{what}** — its config `{f['config'].split()[0]}` no longer exists, "
                        "so the run cannot be relaunched from an experiment id.")
        elif f["ckpt"].startswith("—"):
            gaps.append(f"- **{what}** — the run has neither `best.ckpt` nor `latest.ckpt`.")
    for c in CAMPAIGNS:
        script = c["cmd"].split()[-1] if c["cmd"].startswith("python") else None
        if script and script.endswith(".py") and not (REPO / script).exists():
            gaps.append(f"- **campaign {c['n']}** — `{script}` does not exist.")
    gaps += [
        "- **Seeds are not recorded per run.** The trainer derives them from the run "
        "name, so re-running an experiment gives a different sample. Every number in "
        "the paper is one draw; the campaigns that needed more state their seed count.",
        "- **Peak memory is not recorded per run.** What is recorded is what fitted: "
        "the batch column. `scripts/analysis/vae_smoke_step.py --experiment <id>` "
        "measures the real peak for one step before committing hours to a run.",
    ]
    L.extend(gaps)
    W("")

    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    n_rows = len(PAPER_RUNS) + len(SCRIPT_RUNS) + len(CAMPAIGNS)
    print(f"wrote {out}  ({n_rows} rows: {len(PAPER_RUNS)} runs, "
          f"{len(SCRIPT_RUNS)} baselines, {len(CAMPAIGNS)} campaigns; {len(gaps)} gaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
