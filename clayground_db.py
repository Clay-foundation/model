from pathlib import Path

import geopandas as gpd
import lancedb
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

# To download the existing embeddings run aws s3 sync
# aws s3 sync s3://clay-vector-embeddings/v001/ /my/dir/clay-vector-embeddings-v001
vector_dir = Path("/my/local/directory/clay-vector-embeddings-v001/")

# Create new DB structure or open existing
db = lancedb.connect("embeddings")

# Read all vector embeddings into a list
data = []
for mgrs_tile in vector_dir.glob("*.gpq"):
    print(mgrs_tile)
    tile_df = gpd.read_parquet(mgrs_tile)

    tile_df["tile"] = tile_df["source_url"].apply(lambda x: Path(x).parent.parent.name)  # noqa: E501
    tile_df["idx"] = tile_df["source_url"].apply(lambda x: Path(x).stem.rsplit("_")[-1])  # noqa: E501

    for _, row in tile_df.iterrows():
        data.append(
            {
                "vector": row["embeddings"],
                "path": row["source_url"],
                "tile": row["tile"],
                "date": row["date"],
                "year": row["date"].year,
                "idx": int(row["idx"]),
            }
        )

# Show table names
db.table_names()

# Drop existing table if exists
db.drop_table("clay-v001")

# Create embeddings table and insert the vector data
tbl = db.create_table("clay-v001", data=data, mode="overwrite")


# Visualize some image chips
def plot(df, cols=10):
    fig, axs = plt.subplots(1, cols, figsize=(20, 10))

    for ax, (i, row) in zip(axs.flatten(), df.iterrows()):
        path = row["path"]
        chip = rio.open(path)
        tile = row["tile"]
        idx = row["idx"]
        ax.imshow(
            np.clip(
                (chip.read(indexes=(3, 2, 1)).transpose(1, 2, 0) / 10_000) * 3, 0, 1
            )
        )
        ax.set_title(f"{tile}/{idx}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# Select a vector by index, and search 10 similar pairs, and plot
v = tbl.to_pandas()["vector"].values[2054]
result = tbl.search(query=v).limit(10).to_pandas()
plot(result)
