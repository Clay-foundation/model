import torch

from src.datamodule import ClayDataModule

# Load the pre-trained Clay encoder model
clay_encoder = torch.export.load("checkpoints/compiled/encoder.pt").module()


def load_batch():
    # Initialize the data module with appropriate parameters
    dm = ClayDataModule(
        data_dir="/home/ubuntu/data",
        size=256,
        metadata_path="configs/metadata.yaml",
        batch_size=1,
        num_workers=1,
    )

    # Setup the data module for the 'fit' stage
    dm.setup(stage="fit")
    metadata = dm.metadata

    # Get the training data loader and create an iterator
    trn_dl = dm.train_dataloader()
    iter_dl = iter(trn_dl)

    return iter_dl, metadata


def prepare_data(sensor, metadata, device):
    """
    Load data from the sensor and transfer it to the specified device.

    Args:
    - sensor (dict): Sensor data containing 'pixels', 'time', 'latlon', and 'platform'.
    - metadata (dict): Metadata information for different platforms.
    - device (torch.device): The device to which the data should be transferred.

    Returns:
    - tuple: Transferred cube, timestep, latlon, waves, and gsd tensors.
    """
    cube = sensor["pixels"]
    timestep = sensor["time"]
    latlon = sensor["latlon"]
    platform = sensor["platform"][0]

    # Get wavelengths and ground sampling distance (gsd) from metadata
    waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))
    gsd = torch.tensor([metadata[platform].gsd])

    # Transfer data to the specified device
    cube, timestep, latlon, waves, gsd = map(
        lambda x: x.to(device), (cube, timestep, latlon, waves, gsd)
    )
    return cube, timestep, latlon, waves, gsd


def main():
    dl, metadata = load_batch()

    # Fetch samples from the data loader
    l8_c2l1 = next(dl)
    l8_c2l2 = next(dl)
    linz = next(dl)
    naip = next(dl)
    s1 = next(dl)
    s2 = next(dl)

    # Perform inference with the Clay encoder model
    with torch.no_grad():
        for sensor in (l8_c2l1, l8_c2l2, linz, naip, s1, s2):
            # Load data and transfer to GPU
            batch = prepare_data(sensor, metadata, torch.device("cuda"))

            # Get patch embeddings from the encoder model
            patch_embeddings, *_ = clay_encoder(*batch)

            # Extract the class (CLS) embedding
            cls_embedding = patch_embeddings[:, 0, :]

            # Print the platform and the shape of the CLS embedding
            print(sensor["platform"][0], cls_embedding.shape)


if __name__ == "__main__":
    main()
