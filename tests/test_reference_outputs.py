"""File names and report parsing of the reference notebook outputs that build_split_v4 reads."""

from pathlib import Path

from poregen.dataset.io import discover_volumes, read_reference_report, reference_outputs

REPORT = """Parameter set: batch
Installed from {"url": "file:///x", "vcs_info": {"commit_id": "65c1eea6964afd5f7d2a03636ddb7b94370d1050"}}
Front wall slice: 12
Back wall slice: 233
   - window_size (sauvola_radius): 15
   - k: 0.2
   - min_size: 8 voxels (3D connected components)
"""


def test_names_follow_the_notebook_suffixes_with_the_parameters(tmp_path: Path):
    out = reference_outputs(tmp_path / "Na_10_1_volume_eq_aligned.tif", "batch")
    d = tmp_path / "onlypores files"
    assert out["onlypores"] == d / "Na_10_1_volume_eq_aligned_onlypores_r15_k0.2_min8.tif"
    assert out["samplemask"] == d / "Na_10_1_volume_eq_aligned_samplemask_r15_k0.2_min8.tif"
    assert out["report"] == d / "Na_10_1_volume_eq_aligned_report_r15_k0.2_min8.txt"
    assert reference_outputs(tmp_path / "v.tif", "ipynb")["onlypores"].name == "v_onlypores_r30_k0.125_min8.tif"


def test_report_gives_walls_and_parameters(tmp_path: Path):
    p = tmp_path / "r.txt"
    p.write_text(REPORT)
    assert read_reference_report(p) == {
        "frontwall": 12, "backwall": 233, "sauvola_radius": 15, "sauvola_k": 0.2,
        "min_size_filtering": 8, "reference_commit": "65c1eea6964afd5f7d2a03636ddb7b94370d1050"}


def test_report_keeps_a_wall_the_reference_did_not_find(tmp_path: Path):
    """aligner.find_frontwall returns -1 when no flat region exists; onlypores then excludes nothing."""
    p = tmp_path / "r.txt"
    p.write_text(REPORT.replace("Front wall slice: 12", "Front wall slice: -1"))
    assert read_reference_report(p)["frontwall"] == -1


def test_discovery_skips_the_reference_outputs(tmp_path: Path):
    src = tmp_path / "MedidasDB"
    (src / "onlypores files").mkdir(parents=True)
    (src / "a_volume.tif").write_bytes(b"")
    (src / "onlypores files" / "a_volume_onlypores_r30_k0.125_min8.tif").write_bytes(b"")
    assert [v.volume_id for v in discover_volumes(tmp_path)] == ["MedidasDB__a_volume"]
