"""
Fine-tuning modules for Clay Foundation Model.

This subpackage contains modules for fine-tuning the Clay model on various downstream tasks
including classification, segmentation, regression, and embedding extraction.
"""

try:
    # Classification
    from .classify.eurosat_datamodule import EuroSATDataModule
    from .classify.eurosat_model import EuroSATClassifier
    
    # Segmentation  
    from .segment.chesapeake_datamodule import ChesapeakeDataModule
    from .segment.chesapeake_model import ChesapeakeSegmentor
    
    # Regression
    from .regression.biomasters_datamodule import BioMastersDataModule
    from .regression.factory import Regressor
    
    __all__ = [
        "EuroSATDataModule",
        "EuroSATClassifier", 
        "ChesapeakeDataModule",
        "ChesapeakeSegmentor",
        "BioMastersDataModule", 
        "Regressor",
    ]
except ImportError:
    # Allow subpackage to be imported even if specific components fail
    __all__ = []
