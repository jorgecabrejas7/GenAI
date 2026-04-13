from pathlib import Path

import pytest

from poregen.configs.config import load_config


def test_load_config_derives_dataset_root_from_split_version(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
model:
  name: conv_noattn
loss:
  xct_loss_type: charbonnier
training:
  total_steps: 1
data:
  split_version: v2
"""
    )

    cfg = load_config(cfg_path)

    assert cfg["data"]["split_version"] == "v2"
    assert cfg["data"]["dataset_root"] == "split_v2"


def test_load_config_rejects_inconsistent_split_metadata(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
model:
  name: conv_noattn
loss:
  xct_loss_type: charbonnier
training:
  total_steps: 1
data:
  split_version: v2
  dataset_root: split_v1
"""
    )

    with pytest.raises(ValueError, match="Config mismatch"):
        load_config(cfg_path)
