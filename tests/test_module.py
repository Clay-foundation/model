"""Test ClayMAEModule training orchestration."""

from importlib.resources import files
from unittest.mock import MagicMock

import torch
from torch import nn

from claymodel.module import ClayMAEModule


def _bundled_metadata_path():
    return str(files("claymodel").joinpath("configs/metadata.yaml"))


def _make_module():
    return ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.75,
        shuffle=True,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )


def test_configure_optimizers_structure():
    module = _make_module()
    result = module.configure_optimizers()

    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert result["lr_scheduler"]["interval"] == "step"


def test_configure_optimizers_uses_hparams():
    module = ClayMAEModule(
        model_size="tiny",
        lr=1e-3,
        wd=0.1,
        b1=0.85,
        b2=0.9,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )
    result = module.configure_optimizers()
    optimizer = result["optimizer"]

    pg = optimizer.param_groups[0]
    assert abs(pg["lr"] - 1e-3) < 1e-9
    assert abs(pg["weight_decay"] - 0.1) < 1e-9
    assert pg["betas"] == (0.85, 0.9)


def test_on_train_epoch_start_sets_teacher_eval():
    module = _make_module()
    module.model.teacher.train()
    assert module.model.teacher.training

    module.on_train_epoch_start()
    assert not module.model.teacher.training


class _FakeModel(nn.Module):
    """Fake model that returns fixed loss values."""

    def __init__(self, loss, rec, rep):
        super().__init__()
        self._loss = loss
        self._rec = rec
        self._rep = rep
        self.call_args = None

    def forward(self, x):
        self.call_args = x
        return (self._loss, self._rec, self._rep)


def test_forward_delegates_to_model():
    module = _make_module()
    module.eval()

    expected = (torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5))
    fake = _FakeModel(*expected)
    module.model = fake

    datacube = {"pixels": torch.randn(1, 10, 64, 64)}
    result = module(datacube)

    assert fake.call_args is datacube
    assert result == expected


def _make_module_with_fake_model():
    module = _make_module()
    loss = torch.tensor(1.0, requires_grad=True)
    rec = torch.tensor(0.9)
    rep = torch.tensor(0.1)
    module.model = _FakeModel(loss, rec, rep)
    module.log = MagicMock()
    return module


def test_shared_step_returns_loss_and_logs():
    module = _make_module_with_fake_model()

    batch = {"platform": ["sentinel-2-l2a"]}
    result = module.shared_step(batch, batch_idx=0, phase="train")

    assert torch.isfinite(result)
    # Should log loss, rec_loss, rep_loss for both phase and phase_platform
    assert module.log.call_count == 6


def test_training_step_uses_train_phase():
    module = _make_module_with_fake_model()

    batch = {"platform": ["sentinel-2-l2a"]}
    module.training_step(batch, batch_idx=0)

    log_calls = module.log.call_args_list
    names = [c.kwargs["name"] for c in log_calls]
    assert all("train" in n for n in names)


def test_validation_step_uses_val_phase():
    module = _make_module_with_fake_model()

    batch = {"platform": ["sentinel-2-l2a"]}
    module.validation_step(batch, batch_idx=0)

    log_calls = module.log.call_args_list
    names = [c.kwargs["name"] for c in log_calls]
    assert all("val" in n for n in names)
