import streamlit as st
import numpy as np
import lancedb
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio as rio

st.set_page_config(layout="wide")


# Load embeddings
@st.cache_resource()
def connect_to_database():
    db = lancedb.connect("nbs/embeddings")
    tbl = db.open_table("clay-v0")
    return tbl


@st.cache_resource()
def show_samples(_tbl):
    df = _tbl.head(10).to_pandas()
    # sample 100 random rows
    # samples = df.sample(100).to_dict("records")
    samples = df.to_dict("records")

    cols = st.columns(10)
    options = {}
    for idx, sample in enumerate(samples):
        path = sample["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1]).transpose(1, 2, 0) / 3000
        dem_chip = rio.open(path).read(indexes=[13])[0]
        vv, vh = (
            rio.open(path).read(indexes=[11])[0],
            rio.open(path).read(indexes=[12])[0],
        )
        sar_chip = np.stack([vv, vh, vv], axis=-1)
        sar_chip = sar_chip / np.percentile(sar_chip, 98)  # Normalize
        sar_chip = np.power(sar_chip, 0.45)  # Gamma correction
        with cols[idx % 10]:
            st.text(f"{sample['tile']}-{sample['idx']}")
            plt.imshow(rgb_chip)
            plt.axis("off")
            st.pyplot(plt)
        with cols[idx % 10]:
            plt.imshow(sar_chip)
            plt.axis("off")
            st.pyplot(plt)
        with cols[idx % 10]:
            plt.imshow(dem_chip)
            plt.axis("off")
            st.pyplot(plt)

        options[f"{sample['tile']}-{sample['idx']}"] = {
            "vector": sample["vector"],
            "tile": sample["tile"],
            "year": sample["year"],
        }

    return options


# Function to find similar vectors
def find_similar_vectors(tbl, query):
    tile, year = query["tile"], query["year"]
    filter = f"tile != '{tile}' AND year != {year}"
    result = (
        tbl.search(query=query["vector"], vector_column_name="vector")
        .metric("l2")
        .where(filter, prefilter=True)
        .limit(10)
        .to_pandas()
    )
    # st.dataframe(result)
    cols = st.columns(10)
    for idx, row in result.iterrows():
        path = row["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1]).transpose(1, 2, 0) / 3000
        dem_chip = rio.open(path).read(indexes=[13])[0]
        vv, vh = (
            rio.open(path).read(indexes=[11])[0],
            rio.open(path).read(indexes=[12])[0],
        )
        sar_chip = np.stack([vv, vh, vv], axis=-1)
        sar_chip = sar_chip / np.percentile(sar_chip, 98)  # Normalize
        sar_chip = np.power(sar_chip, 0.45)  # Gamma correction
        with cols[idx % 10]:
            st.text(f"{row['tile']}-{row['idx']}")
            plt.imshow(rgb_chip)
            plt.axis("off")
            st.pyplot(plt)
        with cols[idx % 10]:
            plt.imshow(sar_chip)
            plt.axis("off")
            st.pyplot(plt)
        with cols[idx % 10]:
            plt.imshow(dem_chip)
            plt.axis("off")
            st.pyplot(plt)


# Main app
def main():
    st.title("Clayground")

    tbl = connect_to_database()
    options = show_samples(tbl)

    # UI to select an embedding
    selection = st.selectbox("Select a chip", options=options.keys())
    if selection:
        st.text(f"Selected {selection}")

        query = options[selection]
        find_similar_vectors(tbl, query)


if __name__ == "__main__":
    main()
