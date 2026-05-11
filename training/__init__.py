"""Training utilities for Clay Foundation Model."""

import os

import torch


def configure_training_defaults() -> None:
    """Set global defaults for training performance.

    Called by the training entrypoint (trainer.py / LightningCLI).
    Not called during inference — inference users should not have their
    global torch settings modified by importing claymodel.
    """
    torch.set_float32_matmul_precision("medium")
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
