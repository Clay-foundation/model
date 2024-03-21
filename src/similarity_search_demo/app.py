from pathlib import Path

import geopandas as gpd
import lancedb

# import matplotlib.pyplot as plt
from skimage import io
import streamlit as st
import random
import numpy as np


# To download the existing embeddings run aws s3 sync
# aws s3 sync s3://clay-worldcover-embeddings ./clay-worldcover-embeddings

st.set_page_config(layout="wide")


DATA_DIR = Path("./clay-worldcover-embeddings")
TABLE_NAME = "worldcover-2021-v001"
RESULTS_PER_ROW = 6
RESULT_IMAGE_WIDTH = 256

preselected_chips = {
    "fields_with_urban_areas": {
        "year": 2021,
        "bbox": (
            -9953743.588771349,
            4277895.543077875,
            -9948993.95716417,
            4283755.728707017,
        ),
    },
    "fields_with_rivers": {
        "year": 2021,
        "bbox": (
            -10210223.69555905,
            5663179.07671614,
            -10205474.063951872,
            5669929.772701599,
        ),
    },
    "desert": {
        "year": 2021,
        "bbox": (
            -12542292.81468427,
            4143967.549435384,
            -12537543.183077091,
            4149756.925603093,
        ),
    },
    "multi_pivot_agriculture": {
        "year": 2021,
        "bbox": (
            -11426129.386997048,
            3983060.8683606936,
            -11421379.755389867,
            3988768.546861198,
        ),
    },
    "single_pivot_agriculture": {
        "year": 2021,
        "bbox": (
            -12689531.394506842,
            5273491.081122646,
            -12684781.762899661,
            5279961.006529636,
        ),
    },
    "urban_seashore_with_bridge": {
        "year": 2021,
        "bbox": (
            -8476608.158938471,
            4425362.835132731,
            -8471858.527331293,
            4431303.981269149,
        ),
    },
    "rectangular_fields_with_urban_areas": {
        "year": 2021,
        "bbox": (
            -12433051.287719138,
            3875145.8413827084,
            -12428301.65611196,
            3880800.767686216,
        ),
    },
    "urban_areas_with_vegetation": {
        "year": 2021,
        "bbox": (
            -9554774.533768257,
            3673334.9167054254,
            -9550024.902161077,
            3678895.527814893,
        ),
    },
    "airfield_1": {
        "year": 2021,
        "bbox": (
            -9749509.429662624,
            5861055.456912308,
            -9744759.798055444,
            5867958.31391539,
        ),
    },
    "airfield_2": {
        "year": 2021,
        "bbox": (
            -9094060.267871829,
            5087760.990041647,
            -9089310.636264648,
            5094105.667867075,
        ),
    },
    "airfield_3": {
        "year": 2021,
        "bbox": (
            -12613537.288791966,
            4455101.916637609,
            -12608787.657184785,
            4461059.773456038,
        ),
    },
    "airfield_4": {
        "year": 2021,
        "bbox": (
            -13648956.979157135,
            5710541.093923589,
            -13644207.347549954,
            5717327.612268062,
        ),
    },
    "airfield_5": {
        "year": 2021,
        "bbox": (
            -13625208.821121236,
            4520824.504308893,
            -13620459.189514058,
            4526819.751636654,
        ),
    },
    "airfield_6": {
        "year": 2021,
        "bbox": (
            -12822521.07950787,
            4307228.109648672,
            -12817771.447900692,
            4313104.147891071,
        ),
    },
    "dairy_farms": {
        "year": 2021,
        "bbox": (
            -9350540.37465953,
            5043464.783930202,
            -9345790.743052352,
            5049780.389808697,
        ),
    },
    "farmland_with_rivers": {
        "year": 2021,
        "bbox": (
            -10903669.910207285,
            4366132.880415035,
            -10898920.278600104,
            4372041.129423824,
        ),
    },
    "suburban_area": {
        "year": 2021,
        "bbox": (
            -8989568.372513875,
            3279266.1337867663,
            -8984818.740906697,
            3284658.558216243,
        ),
    },
    "urban_area": {
        "year": 2021,
        "bbox": (
            -9122558.057514906,
            3059980.1272329628,
            -9117808.425907727,
            3065287.9219726124,
        ),
    },
    "warehouses": {
        "year": 2021,
        "bbox": (
            -13663205.873978674,
            5703759.727613576,
            -13658456.242371496,
            5710541.093923589,
        ),
    },
    "warehouses_2": {
        "year": 2021,
        "bbox": (
            -13610959.926299697,
            6007171.863638911,
            -13606210.294692516,
            6014191.341104484,
        ),
    },
    "manhattan_1": {
        "year": 2021,
        "bbox": (
            -8239126.578579488,
            4974262.930964391,
            -8234376.946972307,
            4980533.728883917,
        ),
    },
    "manhattan_2": {
        "year": 2021,
        "bbox": (
            -8239126.578579488,
            4967996.154195476,
            -8234376.946972307,
            4974262.930964391,
        ),
    },
    "manhattan_3": {
        "year": 2021,
        "bbox": (
            -8234376.946972307,
            4974262.930964391,
            -8229627.315365128,
            4980533.728883917,
        ),
    },
    "solar_panels_1": {
        "year": 2021,
        "bbox": (
            -12618286.920399146,
            3690024.3263890552,
            -12613537.288791966,
            3695592.5246229414,
        ),
    },
    "solar_panels_2": {
        "year": 2021,
        "bbox": (
            -12623036.552006325,
            3690024.3263890552,
            -12618286.920399146,
            3695592.5246229414,
        ),
    },
    "solar_panels_3": {
        "year": 2021,
        "bbox": (
            -12575540.23593453,
            3628940.2373186396,
            -12570790.604327349,
            3634480.8516152687,
        ),
    },
    "snow_cover_1": {
        "year": 2021,
        "bbox": (
            -11981836.28503707,
            4532818.441395165,
            -11977086.65342989,
            4538820.58087027,
        ),
    },
    "snow_cover_2": {
        "year": 2021,
        "bbox": (
            -11896342.916107837,
            4720525.705997929,
            -11891593.284500655,
            4726638.492320896,
        ),
    },
}


@st.cache_data
def load_vector_data():

    # Read all vector embeddings into a list
    data = []
    vector_data_files = list(Path(DATA_DIR).joinpath("v002/2021/").glob("*.gpq"))
    for i, strip in enumerate(vector_data_files):

        tile_df = gpd.read_parquet(strip).to_crs("epsg:3857")

        for _, row in tile_df.iterrows():
            data.append(
                {"vector": row["embeddings"], "year": 2021, "bbox": row.geometry.bounds}
            )

    return data


@st.cache_resource
def setup_vector_data_table(_data):

    # Create new DB structure or open existing
    vector_database = lancedb.connect(DATA_DIR)

    for table in vector_database.table_names():
        print(f"Dropping existing table: {table}")
        vector_database.drop_table(TABLE_NAME)

    table = vector_database.create_table(TABLE_NAME, data=_data, mode="overwrite")
    print(f"Table loaded with {len(table)} records")

    # st.success("Vector database setup complete!")
    return table


@st.cache_data
def fetch_image(bbox_string):
    url = f"https://services.terrascope.be/wms/v2?SERVICE=WMS&version=1.1.1&REQUEST=GetMap&layers=WORLDCOVER_2021_S2_TCC&BBOX={bbox_string}&SRS=EPSG:3857&FORMAT=image/png&WIDTH=512&HEIGHT=512"
    image = io.imread(url)
    return image


def fetch_and_display_image(chip, width=256, caption=True):
    bbox_string = ",".join([str(dat) for dat in chip["bbox"]])
    img_params = dict(
        image=fetch_image(bbox_string),
        width=width,
    )
    if caption:
        img_params["caption"] = f"BBOX: {bbox_string} YEAR: {chip['year']}"
    st.image(**img_params)


def fetch_random_chip(data):
    st.session_state.rand_input = data[random.randint(0, len(data) - 1)]
    st.session_state.rand_search_results = None


def set_preselected_chip(data):
    # Preselected options don't include the vector
    # so we have to match the bounding box from the
    # preselected option to the record in `data` to
    # retrive the vector
    [search_input] = [
        d
        for d in data
        if d["bbox"] == preselected_chips[st.session_state.preselected_opt]["bbox"]
    ]
    st.session_state.preselected_input = search_input

    st.session_state.preselected_search_results = None


def set_averaged_chip(data):
    positive_chips = [preselected_chips[opt] for opt in st.session_state.positive_opts]
    negative_chips = [preselected_chips[opt] for opt in st.session_state.negative_opts]

    st.session_state.positive_chips = [
        d for d in data if d["bbox"] in [c["bbox"] for c in positive_chips]
    ]
    st.session_state.negative_chips = [
        d for d in data if d["bbox"] in [c["bbox"] for c in negative_chips]
    ]

    vectors = [c["vector"] for c in st.session_state.positive_chips]
    vectors.extend([-1 * c["vector"] for c in st.session_state.negative_chips])

    st.session_state.averaged_input = {
        "vector": np.mean(vectors, axis=0),
        "bbox": [0.0, 0.0, 0.0, 0.0],
    }

    st.session_state.averaged_search_results = None


def run_search(table, data, search_input, metric, num_results, results_location):

    print(
        f"Search input: {search_input['bbox']}, metric: {metric}, num_results: {num_results}, input vector dims: {len(search_input['vector'])}"
    )

    results = (
        table.search(query=search_input["vector"], vector_column_name="vector")
        .metric(metric)
        .limit(num_results)
        .to_list()
    )

    st.session_state[results_location] = results


def display_search_results(results):

    for _i in range(int(len(results) / RESULTS_PER_ROW) + 1):
        for i, col in enumerate(st.columns(RESULTS_PER_ROW, gap="medium")):
            index = i + (_i * RESULTS_PER_ROW)
            if index >= len(results):
                break
            with col:
                print(f"(row:{_i},col:{i}) bbox: {results[index]['bbox']}")
                fetch_and_display_image(
                    results[index], width=RESULT_IMAGE_WIDTH, caption=False
                )


def main():
    st.title("Clay v2 Embeddings Demo")
    data = load_vector_data()
    table = setup_vector_data_table(data)

    tab1, tab2 = st.tabs(["Random Chip", "Vector Arithmetic"])

    with tab1:

        st.button(
            "Fetch random chip", on_click=fetch_random_chip, kwargs={"data": data}
        )

        if st.session_state.get("rand_input"):
            fetch_and_display_image(st.session_state.rand_input, width=512)

            metric = st.selectbox(
                "Distance metric",
                options=["cosine", "euclidean"],
                key="rand_metric",
            )
            num_results = st.number_input(
                "Number of results",
                min_value=1,
                max_value=100,
                value=5,
                key="rand_num_results",
            )

            st.button(
                "Search with this chip!",
                key="random_button",
                on_click=run_search,
                args=(
                    table,
                    data,
                    st.session_state.rand_input,
                    metric,
                    num_results,
                    "rand_search_results",
                ),
            )

        if st.session_state.get("rand_search_results"):
            display_search_results(st.session_state.rand_search_results)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.multiselect(
                "Vectors with POSITIVE features",
                options=preselected_chips.keys(),
                max_selections=5,
                on_change=set_averaged_chip,
                kwargs={"data": data},
                key="positive_opts",
            )
            if st.session_state.get("positive_chips"):
                for chip in st.session_state["positive_chips"]:
                    fetch_and_display_image(chip, width=256, caption=False)

        with col2:
            st.multiselect(
                "Vectors with NEGATIVE features",
                options=preselected_chips.keys(),
                max_selections=5,
                on_change=set_averaged_chip,
                kwargs={"data": data},
                key="negative_opts",
            )
            if st.session_state.get("negative_chips"):
                for chip in st.session_state["negative_chips"]:
                    fetch_and_display_image(chip, width=256, caption=False)

        if st.session_state.get("averaged_input"):
            metric = st.selectbox(
                "Distance metric",
                options=["cosine", "euclidean"],
                key="averaged_metric",
            )
            num_results = st.number_input(
                "Number of results",
                min_value=1,
                max_value=100,
                value=5,
                key="averaged_num_results",
            )

            st.button(
                "Search with these averaged chips!",
                key="averaged_button",
                on_click=run_search,
                args=(
                    table,
                    data,
                    st.session_state.averaged_input,
                    metric,
                    num_results,
                    "averaged_search_results",
                ),
            )

        if st.session_state.get("averaged_search_results"):
            display_search_results(st.session_state.averaged_search_results)


if __name__ == "__main__":
    main()
