"""Tests for run configuration loading and device/dtype resolution."""

from __future__ import annotations

import pytest
import torch

from common.train.loop import TrainConfig, load_run_config, resolve_device, resolve_dtype


def _write(tmp_path, body: str):
    path = tmp_path / "train.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_run_and_schedule_sections_are_both_parsed(tmp_path) -> None:
    path = _write(tmp_path, """
batch_size: 8
grad_accum: 2
max_steps: 100
schedule:
  max_lr: 3.0e-4
  warmup_steps: 10
  max_steps: 100
""")
    train_cfg, schedule_cfg = load_run_config(path)
    assert train_cfg.batch_size == 8
    assert schedule_cfg.max_lr == pytest.approx(3e-4)
    assert schedule_cfg.warmup_steps == 10


def test_unknown_keys_are_rejected_in_both_sections(tmp_path) -> None:
    """A silently ignored typo would make the config file a lie."""
    with pytest.raises(ValueError, match="unknown run keys"):
        load_run_config(_write(tmp_path, "batch_sixe: 8\n"))
    with pytest.raises(ValueError, match="unknown schedule keys"):
        load_run_config(_write(tmp_path, "schedule:\n  max_lrr: 1.0e-4\n"))


def test_real_configs_load_and_agree_with_their_comments() -> None:
    """Both languages' committed configs must parse and be self-consistent."""
    for lang, tokens in (("Hindi", 523_670_863), ("Nepali", 357_473_575)):
        train_cfg, schedule_cfg = load_run_config(f"{lang}/configs/train.yaml")
        assert schedule_cfg.max_steps == train_cfg.max_steps
        assert schedule_cfg.warmup_steps < train_cfg.max_steps
        # max_steps is one epoch, to within a step of rounding.
        epoch = tokens / train_cfg.tokens_per_step(512)
        assert abs(train_cfg.max_steps - epoch) < 1


def test_tokens_per_step_is_the_effective_batch() -> None:
    cfg = TrainConfig(batch_size=16, grad_accum=4)
    assert cfg.tokens_per_step(512) == 32_768


def test_dtype_resolution_never_assumes_bfloat16() -> None:
    """Kaggle's T4s are Turing: fp16 tensor cores, no bf16."""
    cpu = torch.device("cpu")
    assert resolve_dtype("auto", cpu) is torch.float32
    assert resolve_dtype("float16", cpu) is torch.float16
    assert resolve_device("cpu").type == "cpu"
