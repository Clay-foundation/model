from pathlib import Path

import torch
from torch.export import Dim

from src.model import ClayMAEModule

CHECKPOINT_PATH = "checkpoints/clay-v1-base.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def get_data():
    # Load data
    cube = torch.randn(128, 3, 224, 224).to(device)
    time = torch.randn(128, 4).to(device)
    latlon = torch.randn(128, 4).to(device)
    waves = torch.randn(3).to(device)
    gsd = torch.randn(1).to(device)
    return cube, time, latlon, waves, gsd


def load_model():
    module = ClayMAEModule.load_from_checkpoint(CHECKPOINT_PATH)
    encoder = module.model.encoder  # Get the encoder
    encoder = encoder.to(device)  # Move to device
    return encoder


def main():
    # Load data
    cube, time, latlon, waves, gsd = get_data()

    # Load model
    encoder = load_model()

    # Define dynamic shapes for model export
    batch_size = Dim("batch_size", min=2, max=128)  # Define batch size range
    channel_bands = Dim("channel_bands", min=1, max=12)  # Define channel bands range

    dynamic_shapes = {
        "cube": {0: batch_size, 1: channel_bands},
        "time": {0: batch_size},
        "latlon": {0: batch_size},
        "waves": {0: channel_bands},
        "gsd": {0: None},
    }

    # Export model
    exp_compiled_encoder = torch.export.export(
        mod=encoder,
        args=(cube, time, latlon, waves, gsd),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # tensortrt compiled model
    # trt_encoder = torch_tensorrt.dynamo.compile(
    #     exp_compiled_encoder, [cube, time, latlon, waves, gsd]
    # )

    # Save model
    Path("checkpoints/compiled").mkdir(parents=True, exist_ok=True)
    torch.export.save(exp_compiled_encoder, "checkpoints/compiled/encoder.pt")


if __name__ == "__main__":
    main()
