<<<<<<< HEAD
import streamlit as st
import numpy as np
import lancedb
import matplotlib.pyplot as plt
import rasterio as rio
=======
import lancedb
import matplotlib.pyplot as plt
import rasterio as rio
import streamlit as st
>>>>>>> main
from rasterio.plot import show

st.set_page_config(layout="wide")


# Get preferrred chips
def get_unique_chips(tbl):
    chips = [
        {"tile": "17MNP", "idx": "0271", "year": 2023},
        {"tile": "19HGU", "idx": "0033", "year": 2018},
        {"tile": "33NVB", "idx": "0393", "year": 2020},
        {"tile": "21JVJ", "idx": "0100", "year": 2020},
        {"tile": "34KHD", "idx": "0080", "year": 2018},
        {"tile": "19JCF", "idx": "0215", "year": 2023},
        {"tile": "20HMK", "idx": "0100", "year": 2020},
        {"tile": "37MFT", "idx": "0313", "year": 2023},
        {"tile": "49KHR", "idx": "0020", "year": 2017},
        {"tile": "55LBC", "idx": "0075", "year": 2022},
    ]

<<<<<<< HEAD
    filter = " OR ".join(
        [
            f"(tile == '{chip['tile']}' AND idx == '{chip['idx']}') AND year == {chip['year']}"
            for chip in chips
        ]
    )
    result = tbl.search().where(filter, prefilter=True).to_pandas()
=======
    tile_filter = " OR ".join(
        [
            f"(tile == '{chip['tile']}' "
            f"AND idx == '{chip['idx']}') "
            f"AND year == {chip['year']}"
            for chip in chips
        ]
    )
    result = tbl.search().where(tile_filter, prefilter=True).to_pandas()
>>>>>>> main
    return result


# Load embeddings
@st.cache_resource()
def connect_to_database():
    db = lancedb.connect("nbs/embeddings")
    tbl = db.open_table("clay-v001")
    return tbl


@st.cache_resource()
def show_samples(_tbl):
    df = get_unique_chips(_tbl)
    # df = _tbl.head(10).to_pandas()
    # sample 100 random rows
    # samples = df.sample(100).to_dict("records")
    samples = df.to_dict("records")

    cols = st.columns(10)
    options = {}
    for idx, sample in enumerate(samples):
        path = sample["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1])
        rgb_chip = (rgb_chip - rgb_chip.min()) / (rgb_chip.max() - rgb_chip.min())
        with cols[idx % 10]:
            st.caption(f"{sample['tile']}-{sample['date']}-{sample['idx']}")
            show(rgb_chip)
            plt.axis("off")
            st.pyplot(plt)

        options[f"{sample['tile']}-{sample['idx']}"] = {
            "vector": sample["vector"],
            "tile": sample["tile"],
            "year": sample["year"],
        }

    return options


# Function to find similar vectors
@st.cache_data()
def find_similar_vectors(_tbl, query):
    # tile, year = query["tile"], query["year"]
    # filter = f"tile != '{tile}'"
    result = (
        _tbl.search(query=query["vector"], vector_column_name="vector")
        .metric("cosine")
        # .where(filter, prefilter=True)
        .limit(10)
        .to_pandas()
    )
    # st.dataframe(result)
    cols = st.columns(10)
    for idx, row in result.iterrows():
        path = row["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1])
        rgb_chip = (rgb_chip - rgb_chip.min()) / (rgb_chip.max() - rgb_chip.min())
        with cols[idx % 10]:
            st.caption(f"{row['tile']}-{row['date']}-{row['idx']}")
            show(rgb_chip)
            plt.axis("off")
            st.pyplot(plt)


# Main app
def main():
    st.title("Clayground")

    tbl = connect_to_database()
    options = show_samples(tbl)

    # UI to select an embedding
    with st.sidebar:
        selection = st.selectbox("Select a chip", options=options.keys())

        arithmetic = st.toggle("Arithmetic", False)
        if arithmetic:
            multiselect = st.multiselect(
                "Select multiple chips", options=options.keys(), default=[]
            )

        submit = st.button("Submit")

    if submit and not arithmetic:
        query = options[selection]
        find_similar_vectors(tbl, query)

    if submit and arithmetic and len(multiselect) > 1:
        st.write("Selected:", multiselect)
        v1 = options[multiselect[0]]
        v2 = options[multiselect[1]]
        v3 = (v1["vector"] + v2["vector"]) / 2

        find_similar_vectors(tbl, {"vector": v3})


if __name__ == "__main__":
    main()
