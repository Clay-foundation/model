from pathlib import Path

import geopandas as gpd
import lancedb
import matplotlib.pyplot as plt
from skimage import io

# Set working directory
wd = "./"

# To download the existing embeddings run aws s3 sync
# aws s3 sync s3://clay-worldcover-embeddings /my/dir/clay-worldcover-embeddings

vector_dir = Path(wd + "clay-worldcover-embeddings/2020/")

# Create new DB structure or open existing
db = lancedb.connect(wd + "worldcoverembeddings_db")

# Read all vector embeddings into a list
data = []
for strip in vector_dir.glob("*.gpq"):
    print(strip)
    tile_df = gpd.read_parquet(strip).to_crs("epsg:3857")

    for _, row in tile_df.iterrows():
        data.append(
            {"vector": row["embeddings"], "year": 2020, "bbox": row.geometry.bounds}
        )

# Show table names
db.table_names()

# Drop existing table if exists
# db.drop_table("worldcover-2020-v001")

# Create embeddings table and insert the vector data
tbl = db.create_table("worldcover-2020-v001", data=data, mode="overwrite")


# Visualize some image chips
def plot(df, cols=10):
    fig, axs = plt.subplots(1, cols, figsize=(20, 10))

    for ax, (i, row) in zip(axs.flatten(), df.iterrows()):
        bbox = row["bbox"]
        url = f"https://services.terrascope.be/wms/v2?SERVICE=WMS&version=1.1.1&REQUEST=GetMap&layers=WORLDCOVER_2021_S2_TCC&BBOX={','.join([str(dat) for dat in bbox])}&SRS=EPSG:3857&FORMAT=image/png&WIDTH=512&HEIGHT=512"  # noqa: E501
        image = io.imread(url)
        ax.imshow(image)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# Select a vector by index, and search 10 similar pairs, and plot
v = tbl.to_pandas()["vector"].values[5]
result = tbl.search(query=v).limit(5).to_pandas()
plot(result, 5)
