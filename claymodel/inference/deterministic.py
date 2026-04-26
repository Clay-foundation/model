"""Deterministic inference context manager for Clay Foundation Model."""

import contextlib
import os
import random
from collections.abc import Generator

import torch


@contextlib.contextmanager
def DeterministicInference(seed: int = 42) -> Generator[None, None, None]:
    """Context manager for reproducible Clay model inference.

    Sets all random seeds and enables deterministic CUDA operations.
    Note: This may reduce performance. Clay embeddings are slightly
    non-deterministic by default due to CUDA kernel selection and
    float32 matmul precision, but cosine similarity between runs
    is typically > 0.9999.

    Usage:
        from claymodel.inference import DeterministicInference

        with DeterministicInference(seed=42):
            embeddings = model(datacube)

    Args:
        seed: Random seed for reproducibility.
    """
    # Save current state
    old_seed_py = random.getstate()
    old_seed_torch = torch.random.get_rng_state()
    old_cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_matmul = torch.get_float32_matmul_precision()

    try:
        # Set seeds
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Enable deterministic mode
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_float32_matmul_precision("highest")

        yield
    finally:
        # Restore state
        random.setstate(old_seed_py)
        torch.random.set_rng_state(old_seed_torch)
        if old_cuda_states is not None:
            torch.cuda.set_rng_state_all(old_cuda_states)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        if old_cublas is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
        elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.set_float32_matmul_precision(old_matmul)
