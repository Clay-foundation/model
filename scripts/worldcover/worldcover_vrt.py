"""
Generate virtual raster containing all worldcover tiles that are
in a FlatGeoBuf file. The file used here contains all grid tiles
that intersect with the CONUS US.
"""
import geopandas as gpd

# Template for one band from one worldcover composite file.
source_template = """    <ComplexSource>
      <SourceFilename relativeToVRT="1">{filename}</SourceFilename>
      <SourceBand>{band}</SourceBand>
      <SourceProperties RasterXSize="12000" RasterYSize="12000" DataType="UInt16" BlockXSize="1024" BlockYSize="1024" />
      <SrcRect xOff="0" yOff="0" xSize="12000" ySize="12000" />
      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="12000" ySize="12000" />
      <NODATA>0</NODATA>
    </ComplexSource>
"""

grid = gpd.read_file(
    "https://clay-mgrs-samples.s3.amazonaws.com/esa_worldcover_grid_usa.fgb"
)

for year in 2020, 2021:
    bands = []
    for band in range(1, 5):
        data = ""
        for rowid, row in grid.iterrows():
            data += source_template.format(
                band=band,
                filename=row[f"s2_rgbnir_{year}"].replace(
                    "s3://esa-worldcover-s2/",
                    "https://esa-worldcover-s2.s3.amazonaws.com/",
                ),
                xoff=12000 * (125 - int(row.tile[-3:])),
                yoff=12000 * (49 - int(row.tile[1:3])),
            )
        bands.append(data)

    band1, band2, band3, band4 = bands

    # Construct VRT
    rasterXSize = (125 - 67) * 12000
    rasterYSize = (49 - 24) * 12000

    originX = -125
    originY = 50

    vrt = f"""<VRTDataset rasterXSize="{rasterXSize}" rasterYSize="{rasterYSize}">
    <SRS dataAxisToSRSAxisMapping="2,1">GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]</SRS>
    <GeoTransform> {originX},  8.3333333333333331e-05,  0.0000000000000000e+00,  {originY},  0.0000000000000000e+00, -8.3333333333333331e-05</GeoTransform>
    <VRTRasterBand dataType="UInt16" band="1">
        <NoDataValue>0</NoDataValue>
        <Scale>0.0001</Scale>
        {band1}
    </VRTRasterBand>
    <VRTRasterBand dataType="UInt16" band="2">
        <NoDataValue>0</NoDataValue>
        <Scale>0.0001</Scale>
        {band2}
    </VRTRasterBand>
    <VRTRasterBand dataType="UInt16" band="3">
        <NoDataValue>0</NoDataValue>
        <Scale>0.0001</Scale>
        {band3}
    </VRTRasterBand>
    <VRTRasterBand dataType="UInt16" band="4">
        <NoDataValue>0</NoDataValue>
        <Scale>0.0001</Scale>
        {band4}
    </VRTRasterBand>
    <OverviewList resampling="nearest">2 4 8 16</OverviewList>
    </VRTDataset>
    """

    with open(f"scripts/worldcover/worldcover_index_usa_{year}.vrt", "w") as dst:
        dst.write(vrt)
