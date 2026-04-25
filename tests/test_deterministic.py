"""Test deterministic inference context manager (C3 verification)."""

import torch

from claymodel.inference.deterministic import DeterministicInference
from tests.conftest import make_datacube, make_tiny_encoder


def test_deterministic_produces_identical_output():
    encoder = make_tiny_encoder()
    datacube = make_datacube(batch_size=1, channels=6, size=64)

    with DeterministicInference(seed=42):
        with torch.no_grad():
            out1, *_ = encoder(datacube)

    with DeterministicInference(seed=42):
        with torch.no_grad():
            out2, *_ = encoder(datacube)

    assert torch.equal(out1, out2), "Deterministic mode should produce identical output"


def test_deterministic_restores_state():
    old_precision = torch.get_float32_matmul_precision()
    old_benchmark = torch.backends.cudnn.benchmark

    with DeterministicInference(seed=42):
        pass

    assert torch.get_float32_matmul_precision() == old_precision
    assert torch.backends.cudnn.benchmark == old_benchmark


def test_deterministic_different_seeds_differ():
    datacube = make_datacube(batch_size=1, channels=6, size=64)

    with DeterministicInference(seed=42):
        encoder_copy = make_tiny_encoder()
        # Different random weights → different output is expected
        with torch.no_grad():
            out1, *_ = encoder_copy(datacube)

    with DeterministicInference(seed=99):
        encoder_copy2 = make_tiny_encoder()
        with torch.no_grad():
            out2, *_ = encoder_copy2(datacube)

    # Different seeds → different random weights → different output
    assert not torch.equal(out1, out2)
