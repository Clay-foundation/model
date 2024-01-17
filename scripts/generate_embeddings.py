import glob
import os
import sys
import warnings

import lightning as L
import torch
import tqdm

from src.datamodule import ClayDataModule
from src.model_clay import CLAYModule

# Set some environment variables and parameters
torch.set_float32_matmul_precision(precision="medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


def main(mgrs_tiles_path, checkpoint_path):
    # Load MGRS tiles
    mgrs_tiles = open(file=mgrs_tiles_path).read().splitlines()
    mgrs_tiles.sort(key=lambda m: m[2])

    # Setup trainer and load model weights from checkpoint
    trainer = L.Trainer(precision="bf16-mixed", logger=False)
    model: L.LightningModule = CLAYModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )

    # Generate embeddings for each MGRS tile
    for mgrs_tile in (pbar := tqdm.tqdm(iterable=mgrs_tiles)):
        if len(glob.glob(pathname=f"data/embeddings/{mgrs_tile}_*.gpq")) == 0:
            pbar.set_description(desc=f"Processing MGRS Tile {mgrs_tile}")
            datamodule: L.LightningDataModule = ClayDataModule(
                data_dir=f"s3://clay-tiles-02/02/{mgrs_tile}",
                batch_size=32,
                num_workers=16,
            )
            try:
                trainer.predict(model=model, datamodule=datamodule)
            except RuntimeError as err:
                print(f"Processing of MGRS Tile {mgrs_tile} failed because of {err}")
                warnings.warn(message=repr(err))
            except AssertionError as err:
                print(f"Processing of MGRS Tile {mgrs_tile} failed because of {err}")
                warnings.warn(message=repr(err))

    print("All done!")


if __name__ == "__main__":
    # Usage: python scripts/generate_embeddings.py mgrs_tiles.txt checkpoints/epoch=0-step=0.ckpt
    main(sys.argv[1], sys.argv[2])
