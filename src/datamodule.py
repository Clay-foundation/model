"""
LightningDataModule to loads Earth Observation data from <file format> using
<library>.
"""
import lightning as L


# %%
class BaseDataModule(L.LightningDataModule):
    """
    LightningDataModule for loading <file format> files.

    Uses <library>
    """

    def __init__(self, batch_size: int = 32):
        """
        Go from datacubes to 512x512 chips!

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
        raise NotImplementedError

    def train_dataloader(self):
        """
        Loads the data used in the training loop.
        """
        raise NotImplementedError

    def val_dataloader(self):
        """
        Loads the data used in the validation loop.
        """
        raise NotImplementedError
