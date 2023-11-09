"""
Model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""
import lightning as L
import torch


# %%
class BaseLitModule(L.LightningModule):
    """
    Neural network for performing <task> on <dataset>.

    Implemented using Lightning 2.1.
    """

    def __init__(self, lr: float = 0.001):
        """
        Define layers of the <model_name> model.

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.
        """
        super().__init__()

        # Save hyperparameters like lr, weight_decay, etc to self.hparams
        # https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(logger=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Logic for the neural network's training loop.
        """
        raise NotImplementedError

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Optimizing function used to reduce the loss, so that the predicted
        mask gets as close as possible to the groundtruth mask.

        Using the Adam optimizer with a learning rate of 0.001. See:

        - Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic
          Optimization. ArXiv:1412.6980 [Cs]. http://arxiv.org/abs/1412.6980

        Documentation at:
        https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
