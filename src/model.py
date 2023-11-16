"""
Model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""
import lightning as L
import torch
import torchvision
from lightly.models.modules.masked_autoencoder import MAEBackbone, MAEDecoder


# %%
class MAELitModule(L.LightningModule):
    """
    Neural network for performing <task> on <dataset>.

    Implemented using Lightly with Lightning 2.1.
    """

    def __init__(self, lr: float = 0.001, decoder_dim: int = 256):
        """
        Define layers of the <model_name> model.

        |      Encoder/Backbone     |        Decoder/Head          |
        |---------------------------|------------------------------|
        |  Vision Transformer B/32  |  Masked Autoencoder decoder  |

        References:
        - https://docs.lightly.ai/self-supervised-learning/examples/mae.html
        - He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022).
          Masked Autoencoders Are Scalable Vision Learners. 2022 IEEE/CVF
          Conference on Computer Vision and Pattern Recognition (CVPR),
          15979–15988. https://doi.org/10.1109/CVPR52688.2022.01553

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.
        decoder_dim : int
            Size of the decoder tokens. Default is 256.
        """
        super().__init__()

        # Save hyperparameters like lr, weight_decay, etc to self.hparams
        # https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(logger=True)

        # Input Module (Encoder/Backbone). Vision Tranformer (ViT) B_32
        self.vit = torchvision.models.vit_b_32(weights=None)
        self.backbone: torch.nn.Module = MAEBackbone.from_vit(vit=self.vit)

        # Output Module (Decoder/Head). Masked Autoencoder (MAE) Decoder
        self.decoder: torch.nn.Module = MAEDecoder(
            seq_length=self.vit.seq_length,  # 50
            num_layers=1,
            num_heads=16,
            embed_input_dim=self.vit.hidden_dim,  # 768
            hidden_dim=self.hparams.decoder_dim,  # 256
            mlp_dim=self.hparams.decoder_dim * 4,  # 1024
            out_dim=self.vit.patch_size**2 * 3,  # 3072
            dropout=0,
            attention_dropout=0,
        )

        # Loss functions
        self.loss_mse = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
