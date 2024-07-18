import warnings
from pathlib import Path

import torch
from torch.export import Dim

from src.model import ClayMAEModule

warnings.filterwarnings("ignore")

CHECKPOINT_PATH = "checkpoints/clay-v1-base.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHIP_SIZE = 256


def get_data():
    """
    Generate random data tensors for model input.
    """
    cube = torch.randn(128, 3, CHIP_SIZE, CHIP_SIZE).to(DEVICE)
    timestep = torch.randn(128, 4).to(DEVICE)
    latlon = torch.randn(128, 4).to(DEVICE)
    waves = torch.randn(3).to(DEVICE)
    gsd = torch.randn(1).to(DEVICE)
    return cube, timestep, latlon, waves, gsd


def load_model():
    """
    Load the model from a checkpoint and prepare it for evaluation.
    """
    module = ClayMAEModule.load_from_checkpoint(
        CHECKPOINT_PATH, shuffle=False, mask_ratio=0.0
    )
    encoder = module.model.encoder.eval()  # Get the encoder in eval mode
    encoder = encoder.to(DEVICE)  # Move to the appropriate device
    return encoder


def export_model():
    """
    Export the model with dynamic shapes for deployment.
    """
    cube, timestep, latlon, waves, gsd = get_data()
    encoder = load_model()

    # Define dynamic shapes for model export
    batch_size = Dim("batch_size", min=32, max=1200)
    channel_bands = Dim("channel_bands", min=1, max=10)

    dynamic_shapes = {
        "cube": {0: batch_size, 1: channel_bands},
        "time": {0: batch_size},
        "latlon": {0: batch_size},
        "waves": {0: channel_bands},
        "gsd": {0: None},
    }

    # Export model
    ep = torch.export.export(
        mod=encoder,
        args=(cube, timestep, latlon, waves, gsd),
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )

    # Save the exported model
    Path("checkpoints/compiled").mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, "checkpoints/compiled/encoder.pt")


if __name__ == "__main__":
    export_model()
