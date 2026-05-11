"""Test Matryoshka Representation Learning (MRL) module."""

import torch

from claymodel.mrl import MRL, MRLLoss


def test_mrl_forward_shapes():
    dolls = [16, 32, 64]
    mrl = MRL(features=128, dolls=dolls)
    x = torch.randn(4, 64)
    logits = mrl(x)

    assert len(logits) == 3
    for logit in logits:
        assert logit.shape == (4, 128)


def test_mrl_forward_slices_input():
    """Each doll should project from x[:, :doll], not the full input."""
    dolls = [8, 16]
    mrl = MRL(features=32, dolls=dolls)

    x = torch.randn(1, 16)
    logits = mrl(x)

    # Zero out the second half — first doll (size 8) should be unaffected
    x_zeroed = x.clone()
    x_zeroed[:, 8:] = 0
    logits_zeroed = mrl(x_zeroed)

    assert torch.allclose(logits[0], logits_zeroed[0])
    assert not torch.allclose(logits[1], logits_zeroed[1])


def test_mrl_loss_zero_for_identical():
    """When each representation equals the target, cosine sim is 1, loss is ~0."""
    loss_fn = MRLLoss(weights=[1, 1])

    target = torch.randn(2, 64)
    # Pass the same tensor as both representations — cosine sim should be 1
    representations = [target.clone(), target.clone()]
    loss = loss_fn(representations, target)
    assert loss.item() < 0.01


def test_mrl_loss_weighted():
    """Setting a weight to 0 should eliminate that doll's contribution."""
    loss_fn_equal = MRLLoss(weights=[1, 1])
    loss_fn_zeroed = MRLLoss(weights=[0, 1])

    target = torch.randn(2, 64)
    reps = [torch.randn(2, 64), torch.randn(2, 64)]

    loss_equal = loss_fn_equal(reps, target)
    loss_zeroed = loss_fn_zeroed(reps, target)

    # With first weight=0, only second doll contributes
    assert not torch.allclose(loss_equal, loss_zeroed)


def test_mrl_loss_positive():
    """Loss should be positive for non-identical inputs."""
    loss_fn = MRLLoss(weights=[1, 1])
    target = torch.randn(4, 128)
    reps = [torch.randn(4, 128), torch.randn(4, 128)]
    loss = loss_fn(reps, target)
    assert loss.item() > 0
