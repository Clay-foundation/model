from pathlib import Path

import rasterio
from litdata import optimize


def opentif(path):
    with rasterio.open(path) as src:
        return {
            "pixels": src.read(),
            "date": src.tags()["date"],
            "lon": src.lnglat()[0],
            "lat": src.lnglat()[1],
        }


if __name__ == "__main__":
    # Downloaded 25GB of tif tiles from the s3://clay-tiles-02 bucket
    # converted to litdata with the following script.
    # Then uploaded data to s3://clay-tiles-02-litdata

    wd = Path("/home/tam/Desktop/clay-tiles-02/02")
    out = "/home/tam/Desktop/clay-tiles-02-litdata-optimized"
    Path(out).mkdir()

    paths = sorted(wd.glob("**/*.tif"))
    optimize(
        fn=opentif,  # The function applied over each input.
        inputs=paths,  # Provide any inputs. The fn is applied on each item.
        output_dir=out,  # The directory where the optimized data are stored.
        num_workers=16,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="128MB",  # The maximum number of bytes to write into a data chunk.
        compression="zstd",
    )

# aws s3 sync /home/tam/Desktop/clay-tiles-02-litdata-optimized s3://clay-tiles-02-litdata
