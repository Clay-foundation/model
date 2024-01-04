import lancedb
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import streamlit as st

st.set_page_config(layout="wide")


# Get preferrred chips
def get_unique_chips(tbl):
    # 05WPP-1064 - Straight road line
    # 05WNS-855 - Tree pattern
    # 32TLP-1190 - Urban
    # 32TLP-1272 - SAR Cut
    # 05VPJ-1589 - Good DEM
    # 06WWS-1109 - Cross Roads
    # 05WNS-957
    # 03UUV-1134
    # 06KTF-383
    # 05VPK-395
    # 06VVP-539
    chips = [
        # {"tile": "05WPP", "idx": 1064},
        {"tile": "05WNS", "idx": 855},
        {"tile": "32TLP", "idx": 1190},
        {"tile": "32TLP", "idx": 1272},
        {"tile": "05VPJ", "idx": 1589},
        {"tile": "06WWS", "idx": 1109},
        {"tile": "05WNS", "idx": 957},
        {"tile": "03UUV", "idx": 1134},
        {"tile": "06VUN", "idx": 57},
        # {"tile": "05VPK", "idx": 395},
        {"tile": "06VVP", "idx": 539},
        {"tile": "06VUN", "idx": 173},
    ]
    filter = " OR ".join(
        [f"(tile == '{chip['tile']}' AND idx == {chip['idx']})" for chip in chips]
    )
    result = tbl.search().where(filter, prefilter=True).to_pandas()
    return result


# Load embeddings
@st.cache_resource()
def connect_to_database():
    db = lancedb.connect("nbs/embeddings")
    tbl = db.open_table("clay-v0")
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
        # with cols[idx % 10]:
        #     plt.imshow(sar_chip)
        #     plt.axis("off")
        #     st.pyplot(plt)
        # with cols[idx % 10]:
        #     plt.imshow(dem_chip)
        #     plt.axis("off")
        #     st.pyplot(plt)

        options[f"{sample['tile']}-{sample['idx']}"] = {
            "vector": sample["vector"],
            "tile": sample["tile"],
            "year": sample["year"],
        }

    return options


# Function to find similar vectors
def find_similar_vectors(tbl, query):
    # tile, year = query["tile"], query["year"]
    # filter = f"tile != '{tile}'"
    result = (
        tbl.search(query=query["vector"], vector_column_name="vector")
        .metric("cosine")
        # .where(filter, prefilter=True)
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
        # with cols[idx % 10]:
        #     plt.imshow(sar_chip)
        #     plt.axis("off")
        #     st.pyplot(plt)
        # with cols[idx % 10]:
        #     plt.imshow(dem_chip)
        #     plt.axis("off")
        #     st.pyplot(plt)


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
