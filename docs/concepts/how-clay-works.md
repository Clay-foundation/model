# How Clay Works

Clay v1.5 is a sensor-agnostic geospatial foundation model based on a Masked Autoencoder (MAE) with a DINOv2 teacher. It processes satellite imagery from any sensor and produces dense vector embeddings that capture spatial, spectral, and temporal information about Earth's surface.

## Architecture Overview

Clay v1.5 has four main components:

1. **Dynamic Embedding Block** — Generates patch embeddings from any combination of spectral bands using wavelength-conditioned convolutions (inspired by [DOFA](https://arxiv.org/abs/2403.15356)).

2. **Position Encoding** — Encodes spatial location (lat/lon), time (week/hour), and ground sampling distance (GSD) into the patch representations.

3. **Masked Autoencoder (MAE)** — A Vision Transformer encoder-decoder that reconstructs masked patches. During training, 75% of patches are masked. The reconstruction loss accounts for **90%** of the total training loss.

4. **DINOv2 Teacher** — A frozen DINOv2 model (`vit_large_patch14_reg4_dinov2.lvd142m`) that provides representation targets. The representation loss (cosine similarity) accounts for **10%** of the total loss.

### Model Sizes

| Variant | Encoder dim | Depth | Heads | Encoder params |
|---------|-------------|-------|-------|---------------|
| Tiny    | 192         | 6     | 4     | ~5M           |
| Small   | 384         | 6     | 6     | ~22M          |
| Base    | 768         | 12    | 12    | ~86M          |
| Large   | 1024        | 24    | 16    | ~311M         |

Clay v1.5 was trained with the **Large** variant. The full model is 632M parameters (311M encoder + 15M decoder + 304M frozen teacher).

## Dynamic Embedding (Sensor-Agnostic Input)

Clay can process data from **any satellite sensor** because it uses wavelength-conditioned convolutions rather than fixed band positions. The Dynamic Embedding block:

1. Takes the central wavelength (in micrometers) of each input band
2. Generates per-band convolutional kernels conditioned on those wavelengths
3. Produces patch embeddings that encode both spatial and spectral information

This means you can feed Clay data from Sentinel-2 (10 bands), NAIP (4 bands), or a custom sensor — as long as you provide the correct wavelengths and normalization statistics.

**Key**: Wavelengths must be in **micrometers** (the unit used during training). Do not convert to nanometers.

## Clouds and Shadows

Clay was trained **with** clouds and shadows present in the data. They are not corruption — they produce valid, meaningful embeddings.

- Cloud embeddings encode "this area is cloudy at this time"
- Shadow embeddings encode the presence and characteristics of shadows
- This is by design: the model learns to represent the full range of Earth observation conditions

**Recommendation**: Filter clouds **downstream** if your task requires clear-sky data. Use `PatchAnalyzer.cloud_fraction()` (Sentinel-2 SCL band) to estimate cloud coverage per patch.

## SAR Encoding

Sentinel-1 SAR has two polarizations (VV and VH) that share the same physical wavelength (~5.6 cm, C-band). Since Clay's architecture differentiates bands solely by wavelength, it cannot distinguish VV from VH using the real wavelength.

**Solution**: Synthetic positional wavelengths assigned during training:
- **VV**: 3.5 micrometers
- **VH**: 4.0 micrometers

These values are arbitrary but **fixed** — they must be used exactly as specified in `metadata.yaml` for inference. Changing them will produce incorrect embeddings.

**SAR preprocessing**: Input should be in **dB scale** (`10 * log10(linear_power)`). The `embed()` API handles this conversion automatically when `sensor="sentinel-1-rtc"`.

## Non-Optical Inputs (Advanced)

DEMs, weather data, and other non-optical sources can be processed through Clay by assigning synthetic wavelengths outside the optical range. This is an advanced technique with caveats:

- The model was **not trained** on these data types
- Embedding quality depends on how well the synthetic wavelengths differentiate the input bands
- Results are experimental — validate for your specific use case
- Follow the SAR convention: assign distinct, fixed wavelengths for each input variable

## Temporal Model

Clay v1.5 processes each timestep **independently** — there is no cross-frame attention. Each chip at each date produces its own embedding.

For temporal analysis:
- Generate embeddings for the same location at multiple dates
- Compare embeddings using cosine similarity, PCA, or a trained downstream model
- Large cosine distance between dates indicates significant surface change

**Future**: Clay v2 will add temporal reasoning with cross-frame attention ([#369](https://github.com/Clay-foundation/model/issues/369)).

## Sparse Patches (Raster Edges)

At raster edges, chips may be partially filled with nodata pixels. Clay handles this through its masking mechanism, but embedding quality degrades with too many missing pixels.

**Recommendation**: Use `PatchAnalyzer.valid_fraction()` to assess chip quality and skip chips with `valid_fraction < 0.5`. The CLI supports this via `clay embed --min-valid 0.5`.

## Normalization

Clay uses **per-band z-score normalization**: `(pixel - mean) / std` using statistics computed from the training data. These statistics are stored in `metadata.yaml` for each supported sensor.

Important notes:
- Compute spectral indices (NDVI, EVI, etc.) on **raw data before normalization**
- Embeddings encode richer information than any single spectral index
- Use `normalize()` from the API, or let `embed()` handle it automatically
- For new sensors, compute mean/std from a representative sample of your data

```python
from claymodel import normalize
normalized = normalize(raw_pixels, "sentinel-2-l2a")
```

## Matryoshka Representation Learning (MRL)

MRL was used for approximately 90% of the v1.5 training run (epochs 1–70) before being replaced with a direct linear projection. This means the trained embeddings retain some MRL properties:

- Embeddings can be **truncated** to smaller dimensions (e.g., first 256 of 1024 dims) with graceful quality degradation
- Full 1024-dimensional embeddings are recommended for best quality
- The MRL code is retained in `mrl.py` for reference but is not used in the active model

## Input Format

The model expects a **datacube dictionary** with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `pixels` | `[B, C, H, W]` | Normalized pixel values |
| `time` | `[B, 4]` | `(week_sin, week_cos, hour_sin, hour_cos)` — zeros if unknown |
| `latlon` | `[B, 4]` | `(lat_sin, lat_cos, lon_sin, lon_cos)` — zeros if unknown |
| `gsd` | scalar | Ground sampling distance in meters |
| `waves` | `[C]` | Central wavelength per band in micrometers |

The `embed()` API constructs this datacube automatically from raw input.

## Output Format

The encoder returns a tuple `(encoded, unmasked_indices, masked_indices, mask_matrix)`.

For inference (mask_ratio=0):
- `encoded`: `[B, 1 + num_patches, dim]` — first token is CLS, rest are patch tokens
- **CLS token** (`encoded[:, 0, :]`): Global embedding for the entire chip — shape `[B, 1024]`
- **Patch tokens** (`encoded[:, 1:, :]`): Spatially-localized embeddings — for a 256x256 chip with patch_size=8, this gives a 32x32 grid of 1024-dim vectors

Use the CLS token for chip-level tasks (classification, retrieval, similarity). Use patch tokens for pixel-level tasks (segmentation, change detection at sub-chip resolution).
