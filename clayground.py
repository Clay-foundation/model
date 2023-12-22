"""
Run this application with the following command

streamlit run clayground.py
"""
import lancedb
import matplotlib.pyplot as plt
import rasterio as rio
import streamlit as st

st.set_page_config(layout="wide")


# Get preferrred chips
def get_unique_chips(tbl):
    chips = [
        {"tile": "49RFM", "idx": 1, "year": 2017},
        {"tile": "49RFM", "idx": 2, "year": 2017},
        {"tile": "49RFM", "idx": 3, "year": 2017},
    ]
    filter = " OR ".join(
        [
            f"(tile == '{chip['tile']}' AND idx == {chip['idx']} AND year == {chip['year']})"  # noqa E501
            for chip in chips
        ]
    )
    result = tbl.search().where(filter, prefilter=True).to_pandas()
    return result


# Load embeddings
@st.cache_resource()
def connect_to_database():
    db = lancedb.connect("embeddings")
    tbl = db.open_table("clay-v001")
    return tbl


@st.cache_resource()
def show_samples(_tbl):
    df = get_unique_chips(_tbl)
    samples = df.to_dict("records")

    cols = st.columns(10)
    options = {}
    for idx, sample in enumerate(samples):
        path = sample["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1]).transpose(1, 2, 0) / 3000
        with cols[idx % 10]:
            st.text(f"{sample['tile']}-{sample['idx']}")
            plt.imshow(rgb_chip)
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
    result = (
        tbl.search(query=query["vector"], vector_column_name="vector")
        .metric("cosine")
        .limit(10)
        .to_pandas()
    )
    cols = st.columns(10)
    for idx, row in result.iterrows():
        path = row["path"]
        rgb_chip = rio.open(path).read(indexes=[3, 2, 1]).transpose(1, 2, 0) / 3000

        with cols[idx % 10]:
            st.text(f"{row['tile']}-{row['idx']}")
            plt.imshow(rgb_chip)
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
