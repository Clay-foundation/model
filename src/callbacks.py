from torch.lightning.callbacks import Callback
from torchvision.transforms import v2


class ProgressiveResizing(Callback):
    def __init__(self):
        self.resize_schedule = {
            0: {"batch_size": 4, "num_workers": 4, "size": 64},
            10: {"batch_size": 2, "num_workers": 2, "size": 128},
            20: {"batch_size": 1, "num_workers": 1, "size": 256},
        }

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="fit")

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="validate")
