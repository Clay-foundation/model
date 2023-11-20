"""
LightningDataModule to loads Earth Observation data from <file format> using
<library>.
"""
import lightning as L
import torch


# %%
class RandomDataset(torch.utils.data.Dataset):
    """
    Torch Dataset that returns tensors of size (12, 256, 256) with random
    values.
    """

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 2048

    def __getitem__(self, idx: int):
        return torch.randn(12, 256, 256)


class BaseDataModule(L.LightningDataModule):
    """
    LightningDataModule for loading <file format> files.

    Uses <library>
    """

    def __init__(self, batch_size: int = 32):
        """
        Go from datacubes to 256x256 chips!

        Parameters
        ----------
        batch_size : int
            Size of each mini-batch. Default is 32.

        Returns
        -------
        datapipe : torchdata.datapipes.iter.IterDataPipe
            A torch DataPipe that can be passed into a torch DataLoader.
        """
        super().__init__()
        self.batch_size: int = batch_size

    def setup(self, stage: str | None = None):
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        self.dataset = RandomDataset()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=self.batch_size
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=self.batch_size
        )
