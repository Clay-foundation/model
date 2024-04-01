"""
Model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""

import os
import re

import geopandas as gpd
import lightning as L
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely
import torch
import transformers

torch.set_float32_matmul_precision(precision="medium")


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
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=512,  # default was 224
            patch_size=64,  # default was 16
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

        return outputs

    def training_step(
        self, batch: dict[str, torch.Tensor | list[str]], batch_idx: int
    ) -> torch.Tensor:
        """
        Logic for the neural network's training loop.

        Reference:
        - https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L948-L1010
        """
        x: torch.Tensor = batch["image"]
        # x: torch.Tensor = torch.randn(32, 13, 512, 512)  # BCHW

        # Forward encoder
        outputs_encoder: dict = self(x)
        assert outputs_encoder.last_hidden_state.shape == torch.Size([self.B, 17, 768])
        assert outputs_encoder.ids_restore.shape == torch.Size([self.B, 64])
        assert outputs_encoder.mask.shape == torch.Size([self.B, 64])

        # Forward decoder
        outputs_decoder: dict = self.vit.decoder.forward(
            hidden_states=outputs_encoder.last_hidden_state,
            ids_restore=outputs_encoder.ids_restore,
        )
        # output shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        assert outputs_decoder.logits.shape == torch.Size([self.B, 64, 53248])

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
            batch_size=self.B,
        )

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor | list[str]], batch_idx: int
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        x: torch.Tensor = batch["image"]
        # x: torch.Tensor = torch.randn(32, 13, 512, 512)  # BCHW

        # Forward encoder
        outputs_encoder: dict = self(x)
        assert outputs_encoder.last_hidden_state.shape == torch.Size([self.B, 17, 768])
        assert outputs_encoder.ids_restore.shape == torch.Size([self.B, 64])
        assert outputs_encoder.mask.shape == torch.Size([self.B, 64])

        # Forward decoder
        outputs_decoder: dict = self.vit.decoder.forward(
            hidden_states=outputs_encoder.last_hidden_state,
            ids_restore=outputs_encoder.ids_restore,
        )
        # output shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        assert outputs_decoder.logits.shape == torch.Size([self.B, 64, 53248])

        # Log training loss and metrics
        loss: torch.Tensor = self.vit.forward_loss(
            pixel_values=x, pred=outputs_decoder.logits, mask=outputs_encoder.mask
        )
        self.log(
            name="val/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.B,
        )

        return {"loss": loss, "logits": outputs_decoder.logits}

    def predict_step(
        self, batch: dict[str, torch.Tensor | list[str]], batch_idx: int
    ) -> gpd.GeoDataFrame:
        """
        Logic for the neural network's prediction loop.

        Takes batches of image inputs, generate the embeddings, and store them
        in a geopandas.GeoDataFrame with spatiotemporal metadata.

        Steps:
        1. Image inputs are passed through the encoder model to produce raw
           embeddings of shape (B, 65, 768), where B is the batch size, 65 is
           the dimension that consists of 1 cls_token + 64 patch embeddings
           (that were flattened from the original 8x8 grid), and 768 is the
           embedding length.
        2. Taking only the (B, 64, 768) patch embeddings, we compute the mean
           along the 64-dim, to obtain final embeddings of shape (B, 768).

                                                ______
         cls_token    /        Patch          /      /|
         embeddings  /    +    embeddings    /_____ / |    =>  (1+64, 768)
         (1, 768)   /          (8x8, 768)   |      |  |        = (65, 768)
                   /           = (64, 768)  |      | /
                                            |______|/
                                                |                            /
                                                --------> Final embedding   /
                           compute mean along spatial dim = (1, 768)       /
                                                                          /

        3. Embeddings are joined with spatiotemporal metadata (date and
           bounding box polygon) in a geopandas.GeoDataFrame table. The
           coordinates of the bounding box are in an OGC:CRS84 projection (i.e.
           longitude/latitude). The table is as follows:

           |  source_url  |    date    |      embeddings      |   geometry   |
           |--------------|------------|----------------------|--------------|
           | s3://1A.tif  | 2021-01-01 | [0.1, 0.4, ... x768] | POLYGON(...) |
           | s3://2B.tif  | 2021-06-30 | [0.2, 0.5, ... x768] | POLYGON(...) |
           | s3://3C.tif  | 2021-12-31 | [0.3, 0.6, ... x768] | POLYGON(...) |
        """
        # Get image, bounding box, EPSG code, and date inputs
        x: torch.Tensor = batch["image"]  # image of shape (1, 13, 512, 512) # BCHW
        bboxes: np.ndarray = batch["bbox"].cpu().__array__()  # bounding boxes
        epsgs: torch.Tensor = batch["epsg"]  # coordinate reference systems as EPSG code
        dates: list[str] = batch["date"]  # dates, e.g. ['2022-12-12', '2022-12-12']
        source_urls: list[str] = batch[  # URLs, e.g. ['s3://1.tif', 's3://2.tif']
            "source_url"
        ]

        # Forward encoder
        self.vit.config.mask_ratio = 0  # disable masking
        outputs_encoder: dict = self(x)

        # Get embeddings generated from encoder
        embeddings_raw: torch.Tensor = outputs_encoder.last_hidden_state
        assert embeddings_raw.shape == torch.Size(
            [self.B, 65, 768]  # (batch_size, sequence_length, hidden_size)
        )
        assert not torch.isnan(embeddings_raw).any()  # ensure no NaNs in embedding

        # Take the mean of the embeddings along the sequence_length dimension
        # excluding the first cls token embedding, compute over patch embeddings
        embeddings_mean: torch.Tensor = embeddings_raw[:, 1:, :].mean(dim=1)
        assert embeddings_mean.shape == torch.Size(
            [self.B, 768]  # (batch_size, hidden_size)
        )

        # Create table to store the embeddings with spatiotemporal metadata
        unique_epsg_codes = set(int(epsg) for epsg in epsgs)
        if len(unique_epsg_codes) == 1:  # check that there's only 1 unique EPSG
            epsg: int = batch["epsg"][0]
        else:
            raise NotImplementedError(
                f"More than 1 EPSG code detected: {unique_epsg_codes}"
            )

        gdf = gpd.GeoDataFrame(
            data={
                "source_url": pd.Series(data=source_urls, dtype="string[pyarrow]"),
                "date": pd.to_datetime(arg=dates, format="%Y-%m-%d").astype(
                    dtype="date32[day][pyarrow]"
                ),
                "embeddings": pa.FixedShapeTensorArray.from_numpy_ndarray(
                    embeddings_mean.cpu().detach().__array__()
                ),
            },
            geometry=shapely.box(
                xmin=bboxes[:, 0],
                ymin=bboxes[:, 1],
                xmax=bboxes[:, 2],
                ymax=bboxes[:, 3],
            ),
            crs=f"EPSG:{epsg}",
        )
        gdf = gdf.to_crs(crs="OGC:CRS84")  # reproject from UTM to lonlat coordinates

        return gdf

    def on_predict_epoch_end(self) -> gpd.GeoDataFrame:
        """
        Logic to gather all the results from one epoch in a prediction loop.

        This is where we combine all the vector embeddings generated from each
        mini-batch prediction (stored in geopandas.GeoDataFrame tables) in a
        row-wise manner. The embeddings are then output to GeoParquet files
        according to their MGRS code.

        Steps:
        1. Concatenate all geopandas.GeoDataFrame objects row-wise
        2. Find all unique MGRS names in the big GeoDataFrame.
        3. Split the GeoDataFrame by MGRS name, and output each MGRS-subsetted
           table to a GeoParquet file.
            ________________     ________________     ________________
           |                |   |                |   |                |
           | GeoDataFrame 1 |   | GeoDataFrame 2 |   | GeoDataFrame 3 |
           | for MGRS 12ABC |   | for MGRS 23DEF |   | for MGRS 45GHI |
           |________________|   |________________|   |________________|
                    |                    |                    |
                    v                    v                    v
               12ABC_v01.gpq        23DEF_v01.gpq        45GHI_v01.gpq
        """
        # Combine list of geopandas.GeoDataFrame objects
        results: list[gpd.GeoDataFrame] = self.trainer.predict_loop.predictions
        if results:
            gdf: gpd.GeoDataFrame = pd.concat(
                objs=results, axis="index", ignore_index=True
            )
        else:
            print(
                "No embeddings generated, "
                f"possibly no GeoTIFF files in {self.trainer.datamodule.data_dir}"
            )
            return

        # Save embeddings in GeoParquet format, one file for each MGRS code
        outfolder: str = f"{self.trainer.default_root_dir}/data/embeddings"
        os.makedirs(name=outfolder, exist_ok=True)

        # Find unique MGRS names (e.g. '12ABC'), e.g.
        # from 's3://.../.../claytile_12ABC_20201231_v02_0001.tif', get 12ABC
        mgrs_codes = gdf.source_url.str.split("/").str[-1].str.split("_").str[1]
        unique_mgrs_codes = mgrs_codes.unique()
        for mgrs_code in unique_mgrs_codes:
            if re.match(pattern=r"(\d{2}[A-Z]{3})", string=mgrs_code) is None:
                raise ValueError(
                    "MGRS code should have 2 numbers and 3 letters (e.g. 12ABC), "
                    f"but got {mgrs_code} instead"
                )

            # Subset GeoDataFrame to a single MGRS code
            _gdf: gpd.GeoDataFrame = gdf.loc[mgrs_codes == mgrs_code].reset_index()

            # Get min/max date from GeoDataFrame
            minmax_date: pd.Series = _gdf.date.agg(func=["min", "max"])
            min_date: str = minmax_date["min"].strftime("%Y%m%d")
            max_date: str = minmax_date["max"].strftime("%Y%m%d")

            # Output to a GeoParquet filename like
            # {MGRS:5}_{MINDATE:8}_{MAXDATE:8}_v{VERSION:3}.gpq
            outpath = f"{outfolder}/{mgrs_code}_{min_date}_{max_date}_v001.gpq"
            _gdf.to_parquet(path=outpath, compression="ZSTD", schema_version="1.0.0")
            print(
                f"Saved {len(_gdf)} rows of embeddings of "
                f"shape {gdf.embeddings.iloc[0].shape} to {outpath}"
            )

        return gdf

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
