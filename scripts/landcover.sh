wd=/datadisk

# Download Worldcover layer
aws s3 sync s3://esa-worldcover/v200/2021/map $wd/esa-worldcover-v200-2021-map --no-sign-request

# Download MGRS grid kml and convert to fgb
curl -o $wd/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml
ogr2ogr \
    -overwrite\
    $wd/mgrs.fgb\
    $wd/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml \
    -nlt multipolygon\
    -sql "select Name, ExtractMultiPolygon(geometry) as geometry from Features"\
    -dialect sqlite

# Reduce reolution to 40m and merge
counter=0
for file in $wd/esa-worldcover-v200-2021-map/*.tif; do
    counter=$((counter+1))
    echo "$counter $file"
    gdal_translate -ovr 3 $file $wd/tmp/tmp_$counter.tif
done

gdal_merge.py -o $wd/worldcover.tif $wd/tmp/*.tif

# Intersect worldcover with mgrs and sample based on landcover
python landcover.py --wd=$wd --worldcover=$wd/worldcover.tif --mgrs=$wd/mgrs.fgb
