#!/bin/bash

DIR='/opt/intel/openvino'
ALL=false

for v in "$@"
do
    if [ "$v" = '-a' ]; then
        ALL=true
    else
        DIR=$v
    fi
done

echo $ALL
echo $DIR

if [ "$ALL" = true ]; then
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 -o models
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o models
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models
else
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 -o models --precisions FP32-INT1
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o models --precisions FP16-INT8
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models --precisions FP16-INT8
    python3 $DIR/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models --precisions FP16-INT8
fi