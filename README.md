# Computer Pointer Controller

This is a fun application that makes use Intel OpenVino toolkit to run inference on an image or video or camera stream of a person, with four models in pipeline to finally extract eye gazing direction of the person and automatically move mouse pointer there.

## Project Set Up and Installation

Project has following directories:
```
.
|--media/
|  |--<sample_media_files>
|
|--src/
|  |--face_detection.py
|  |--facial_landmarks_detection.py
|  |--head_pose_estimation.py
|  |--gaze_estimation.py
|  |--input_feeder.py
|  |--mouse_controller.py
|  |--test_models.py
|  |--main.py
|
|--models/intel/
|        |--face-detection-adas-binary-0001/FP32
|        |                                 |--face-detection-adas-binary-0001.xml
|        |                                 |--face-detection-adas-binary-0001.bin
|        |--landmarks-regression-retail-0009/FP16
|        |                                 |--landmarks-regression-retail-0009.xml
|        |                                 |--landmarks-regression-retail-0009.bin
|        |--head-pose-estimation-adas-0001/FP16
|        |                                 |--head-pose-estimation-adas-0001.xml
|        |                                 |--head-pose-estimation-adas-0001.bin
|        |--gaze-estimation-adas-0002/FP16
|                                          |--gaze-estimation-adas-0002.xml
|                                          |--gaze-estimation-adas-0002.bin
|
|--requirements.txt
```

Models used for making inference are IRs from OpenVino Model Zoo.
These can be found here:
* [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

OpenVino's [Model Downloader](https://docs.openvinotoolkit.org/latest/_tools_downloader_README.html) was used to download all the models.<br>

**Note:** Specifying output directory as `models` to `downloader.py` model downloader script while running command in project root directory, file will automatically create sub-directory structure like above.

### Dependencies
Python 3 virtual environment with pip3 on Ubuntu 18.04 was used to develop the app.<br>

Installation of following pre-requisites are required:
- [Intel OpenVino Toolkit](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
- OpenCV 4 (Included in OpenVino Toolkit)
- Following direct python package dependencies (see setup)
  - Numpy
  - PyAutoGUI

To setup the project follow steps: 
1. Install [virtualenv](https://pypi.org/project/virtualenv/)
```
      pip install virtualenv
```
2. Create a virutal environment
```
      virtualenv -p python3 env
```
3. Activate the created virtual environment
```
      source env/bin/activate
```
4. Install packages listed in `requirements.txt`
```
      pip3 install -r requirements.txt
```

## Demo
After installing all the required dependencies and model files with correct precisions, run following command for a quick demo:
```
python3 src/main.py -t video -i bin/demo.mp4
```

## Documentation

Code base is moduler with each module having seperate concerns:<br>
- `face_detection.py`: Class for utilizing Face Detection model to extract box coordinates of face of the person in frame. These coordinates are used to crop face from frame.
- `facial_landmarks_detection.py`: Class for utilizing Facial Landmarks Detection model to get the facial landmarks coordinates from face. However, for the app only required eye landmarks are returned which are later used to extract left and right eye.
- `head_pose_estimaion.py`: Class for utilizing Head Pose Estimation model to extract, from face, the head pose angles- yaw, pitch and roll as list with indices in order respectively. These angles are later required in pipeline.
- `gaze_estimation.py`: Class for utilizing Gaze Estimation model which given left and right eye images as well as head pose angles, yields the gaze vectors. Gaze vectors define direction of person's gaze.
- `input_feeder.py`: Convenient class for reading and feeding frames from input media.
- `mouse_controller.py`: Convenient class for controlling mouse pointer.
- `test_models.py`: Script written for purpose of individual testing of models for correct output. Appropriate function can be run to check working of model.
- `main.py`: Script, which is the starting point for the app.

Below image demonstrates pipeline of code:<br>
![pipeline](pipeline.png)

Following command can be run in project root directory to execute the app:
  ```
  python3 src/main.py -t <media_type> -i <path_to_input_file>
  ```
Arguments to `main.py`-
- `-t`: (Required) Type of media input, `image`, `video` or `cam`.
- `-i`: Path to input file. Not required only for `cam` type.
- `-r`: Option to visualize the intermediate inference results from models.
- `-d`: Option to select device to run inference on.


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
