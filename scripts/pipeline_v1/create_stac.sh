# Download STAC items to a local folder for indexing
# Run this script in a location where all downloaded items
# can be written to an ./items subfolder.

export AWS_REQUEST_PAYER=requester

rio stac --without-raster s3://naip-source/ma/2021/60cm/rgbir/42070/m_4207001_se_19_060_20211024.tif | jq > items/naip_m_4207001_se_19_060_20211024.json
rio stac --without-raster s3://naip-source/ma/2021/60cm/rgbir/42070/m_4207001_sw_19_060_20211024.tif | jq > items/naip_m_4207001_sw_19_060_20211024.json
rio stac --without-raster s3://naip-source/ma/2021/60cm/rgbir/42070/m_4207002_sw_19_060_20211024.tif | jq > items/naip_m_4207002_sw_19_060_20211024.json
rio stac --without-raster s3://naip-source/ma/2021/60cm/rgbir/42070/m_4207009_ne_19_060_20211024.tif | jq > items/naip_m_4207009_ne_19_060_20211024.json
rio stac --without-raster s3://naip-source/ma/2021/60cm/rgbir/42070/m_4207009_nw_19_060_20211024.tif | jq > items/naip_m_4207009_nw_19_060_20211024.json

rio stac s3://nz-imagery/auckland/auckland_2010_0.075m/rgb/2193/BA31_1000_0848.tiff | jq > items/nz-auckland-2010-75mm-rgb-2193-BA31_1000_0848.json
rio stac s3://nz-imagery/auckland/auckland_2010_0.075m/rgb/2193/BA32_1000_3212.tiff | jq > items/nz-auckland-2010-75mm-rgb-2193-BA32_1000_3212.json
rio stac s3://nz-imagery/auckland/auckland_2010_0.075m/rgb/2193/BA32_1000_3206.tiff | jq > items/nz-auckland-2010-75mm-rgb-2193-BA32_1000_3206.json
rio stac s3://nz-imagery/auckland/auckland_2010_0.075m/rgb/2193/BA32_1000_3111.tiff | jq > items/nz-auckland-2010-75mm-rgb-2193-BA32_1000_3111.json
rio stac s3://nz-imagery/auckland/auckland_2010_0.075m/rgb/2193/BA32_1000_3410.tiff | jq > items/nz-auckland-2010-75mm-rgb-2193-BA32_1000_3410.json

curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086110_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086110_20240311_20240312_02_T2_SR.json
curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086109_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086109_20240311_20240312_02_T2_SR.json
curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086108_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086108_20240311_20240312_02_T2_SR.json
curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086107_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086107_20240311_20240312_02_T2_SR.json
curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086106_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086106_20240311_20240312_02_T2_SR.json
curl https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/items/LC09_L2SR_086075_20240311_20240312_02_T2_SR | jq > items/landsat-c2l2-sr-LC09_L2SR_086075_20240311_20240312_02_T2_SR.json

curl https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_20HMF_20240309_0_L2A | jq > items/sentinel-2-l2a-S2B_20HMF_20240309_0_L2A.json
curl https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/S2A_T20HNJ_20240311T140636_L2A | jq > items/sentinel-2-l2a-S2A_T20HNJ_20240311T140636_L2A.json
curl https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/S2B_T13RFJ_20240312T173512_L2A | jq > items/sentinel-2-l2a-S2B_T13RFJ_20240312T173512_L2A.json
curl https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_38NLM_20240312_0_L2A | jq > items/sentinel-2-l2a-S2B_38NLM_20240312_0_L2A.json
