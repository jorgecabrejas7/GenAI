"""The author's floor rule for continuing a VAE past its early stop."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fw", REPO / "scripts" / "vae_floor_watch.py")
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)


def row(step, coll, dice=0.93, sharp=0.80, xct=0.037):
    return {"step": step, "kl_collapsed_fraction": coll, "kl_per_channel": 0.2,
            "std_mean": 0.9, "mu_std": 0.4, "dice_pore": dice,
            "sharpness_recon_over_gt": sharp, "xct_loss": xct}


def test_floor_still_dropping_and_nothing_worse_continues():
    rows = [row(12658, 0.50), row(25317, 0.45), row(37975, 0.40)]
    assert not fw.decide(rows, 8)["stop"]


def test_floor_flat_over_two_full_validations_stops():
    d = fw.decide([row(12658, 0.40), row(25317, 0.35), row(37975, 0.40)], 8)
    assert d["stop"] and "did not drop" in d["reasons"][0]


def test_one_bad_row_is_not_enough_two_consecutive_are():
    ok = [row(1, .6), row(2, .5, dice=0.93), row(3, .4, dice=0.915), row(4, .3, dice=0.93)]
    assert not fw.decide(ok, 8)["stop"]
    bad = [row(1, .6), row(2, .5, dice=0.93), row(3, .4, dice=0.915), row(4, .3, dice=0.915)]
    d = fw.decide(bad, 8)
    assert d["stop"] and "dice_pore" in d["reasons"][0]


def test_xct_and_sharpness_guards():
    rows = [row(1, .6), row(2, .5), row(3, .4, xct=0.0385), row(4, .3, xct=0.0385)]
    assert "xct_loss" in fw.decide(rows, 8)["reasons"][0]
    rows = [row(1, .6), row(2, .5), row(3, .4, sharp=0.785), row(4, .3, sharp=0.785)]
    assert "sharpness" in fw.decide(rows, 8)["reasons"][0]


# ── phase 1 must see the trainer end, whatever wraps it ──────────────────────

def test_a_zombie_trainer_counts_as_finished(tmp_path):
    import subprocess, time
    child = subprocess.Popen(["true"])        # exits at once; not reaped yet
    time.sleep(0.5)
    assert fw.pid_state(child.pid) == "Z"
    assert "zombie" in fw.trainer_finished(child.pid, tmp_path)
    child.wait()
    assert fw.trainer_finished(child.pid, tmp_path) == "process gone"


def test_the_run_status_ends_the_wait_even_if_a_process_lingers(tmp_path):
    import json, os
    (tmp_path / "run_metadata.json").write_text(json.dumps({"status": "running"}))
    assert fw.trainer_finished(os.getpid(), tmp_path) is None
    (tmp_path / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    assert fw.trainer_finished(os.getpid(), tmp_path) == "run_metadata status completed"


def test_a_wrapper_pid_is_refused():
    import os
    assert not fw.is_trainer(os.getpid())    # pytest, not train_vae.py
