"""
Clay Foundation Model - An open source AI model and interface for Earth observation.

This package provides tools for working with satellite imagery and Earth
observation data using foundation models built on Vision Transformers.

Example usage:
    from claymodel.datamodule import ClayDataModule
    from claymodel.module import ClayMAEModule

    # Create data module and model
    datamodule = ClayDataModule(...)
    model = ClayMAEModule(...)
"""

__version__ = "1.5.0"

# Main components available for import
__all__ = [
    "ClayMAEModule",
    "ClayDataModule",
    "clay_mae_base",
    "clay_mae_large",
    "clay_mae_small",
    "clay_mae_tiny",
]
