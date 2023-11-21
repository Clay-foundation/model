"""
Model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""
import lightning as L
import torch
import transformers


# %%
class ViTLitModule(L.LightningModule):
    """
    Vision Transformer neural network trained using a Masked Autoencoder setup.

    Implemented using transformers with Lightning 2.1.
    """

    def __init__(self, lr: float = 0.001, mask_ratio: float = 0.75):
        """
        Define layers of the Vision Transformer model.

        |      Encoder/Backbone     |        Decoder/Head          |
        |---------------------------|------------------------------|
        |  Vision Transformer B/32  |  Masked Autoencoder decoder  |

        References:
        - https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/vit_mae
        - He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022).
          Masked Autoencoders Are Scalable Vision Learners. 2022 IEEE/CVF
          Conference on Computer Vision and Pattern Recognition (CVPR),
          15979–15988. https://doi.org/10.1109/CVPR52688.2022.01553

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.
        mask_ratio : float
            The ratio of the number of masked tokens in the input sequence.
            Default is 0.75.
        """
        super().__init__()

        # Save hyperparameters like lr, weight_decay, etc to self.hparams
        # https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(logger=True)

        # Vision Transformer Masked Autoencoder configuration
        config_vit = transformers.ViTMAEConfig(
            hidden_size=768,
            num_hidden_layers=12,
            ntermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=256,  # default was 224
            patch_size=32,  # default was 16
            num_channels=13,  # default was 3
            qkv_bias=True,
            decoder_num_attention_heads=16,
            decoder_hidden_size=512,
            decoder_num_hidden_layers=8,
            decoder_intermediate_size=2048,
            mask_ratio=self.hparams.mask_ratio,
            norm_pix_loss=False,
        )

        # Vision Tranformer (ViT) B_32 (Encoder + Decoder)
        self.vit: torch.nn.Module = transformers.ViTMAEForPreTraining(config=config_vit)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass (Inference/Prediction).
        """
        outputs: dict = self.vit.base_model(x)

        self.B = x.shape[0]
        assert outputs.last_hidden_state.shape == torch.Size([self.B, 17, 768])
        assert outputs.ids_restore.shape == torch.Size([self.B, 64])
        assert outputs.mask.shape == torch.Size([self.B, 64])

        return outputs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Logic for the neural network's training loop.

        Reference:
        - https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L948-L1010
        """
        x: torch.Tensor = batch
        # x: torch.Tensor = torch.randn(32, 13, 256, 256)  # BCHW

        # Forward encoder
        outputs_encoder: dict = self(x)

        # Forward decoder
        outputs_decoder: dict = self.vit.decoder.forward(
            hidden_states=outputs_encoder.last_hidden_state,
            ids_restore=outputs_encoder.ids_restore,
        )
        # output shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        assert outputs_decoder.logits.shape == torch.Size([self.B, 64, 13312])

        # Log training loss and metrics
        loss: torch.Tensor = self.vit.forward_loss(
            pixel_values=x, pred=outputs_decoder.logits, mask=outputs_encoder.mask
        )
        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        pass

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
