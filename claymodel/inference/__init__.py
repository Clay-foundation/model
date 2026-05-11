"""Inference utilities for Clay Foundation Model."""

from claymodel.inference.deterministic import DeterministicInference
from claymodel.inference.masking import PatchAnalyzer

__all__ = ["DeterministicInference", "PatchAnalyzer"]

# Pipeline exports are available when inference extras are installed.
# Importing them here would require rustac/lazycogs/geopandas at package
# import time, so they are accessed via claymodel.inference.pipeline.
