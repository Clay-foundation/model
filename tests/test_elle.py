"""Test ELLE quality scoring (C1 verification)."""

import tempfile

import torch

from claymodel.inference.elle import ELLEProbe


def test_elle_probe_score():
    # Create a probe with known weights
    coef = torch.randn(1024)
    probe = ELLEProbe(coef=coef, intercept=0.5)

    embeddings = torch.randn(4, 1024)
    scores = probe.score(embeddings)

    assert scores.shape == (4,)
    # Verify manual computation
    expected = embeddings @ coef + 0.5
    assert torch.allclose(scores, expected)


def test_elle_probe_save_load():
    coef = torch.randn(192)
    probe = ELLEProbe(coef=coef, intercept=-0.3)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        probe.save(f.name)
        loaded = ELLEProbe.from_file(f.name)

    assert torch.allclose(loaded.coef, coef)
    assert loaded.intercept == -0.3


def test_elle_probe_default_not_found():
    """Default probe should raise FileNotFoundError when weights aren't bundled."""
    try:
        ELLEProbe.default()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected


def test_elle_probe_device_handling():
    coef = torch.randn(192)
    probe = ELLEProbe(coef=coef, intercept=0.0)

    # Score should work with embeddings on same device as coef
    embeddings = torch.randn(2, 192)
    scores = probe.score(embeddings)
    assert scores.shape == (2,)
