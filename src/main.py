from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation

from input_feeder import InputFeeder
from mouse_controller import MouseController

from argparse import ArgumentParser

import cv2

### Model IR paths
path_face_detection = "models/intel/face-detection-adas-binary-0001/FP32/face-detection-adas-binary-0001.xml"
path_facial_landmarks_detection = "models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml"
path_head_pose_estimation = "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
path_gaze_estimation = "models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"

### Model global variables
face_detection = None
facial_landmarks_detection = None
head_pose_estimation = None
gaze_estimation = None

### Offsets to crop eyes from face
x_offset = 50
y_offset = 25

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-t", "--input_type", required=True, type=str, \
        help="Type of input. Valid values are 'image', 'cam', 'video'")
    
    parser.add_argument("-i", "--input", required=False, type=str, \
        help="Path to input file. Required if input type is 'image' or 'video'")

    return parser

### Initiate & load all required models
def init_models(device="CPU"):
    # Using global variables, not defining new variables
    global face_detection
    global facial_landmarks_detection
    global head_pose_estimation
    global gaze_estimation

    face_detection = Face_Detection(path_face_detection, device)
    facial_landmarks_detection = Facial_Landmarks_Detection(path_facial_landmarks_detection, device)
    head_pose_estimation = Head_Pose_Estimation(path_head_pose_estimation, device)
    gaze_estimation = Gaze_Estimation(path_gaze_estimation, device)

    face_detection.load_model()
    facial_landmarks_detection.load_model()
    head_pose_estimation.load_model()
    gaze_estimation.load_model()

### Crop rectangle from given coordinates
def crop_rect(image, coords):
    xmin = coords[0]
    ymin = coords[1]
    xmax = coords[2]
    ymax = coords[3]

    crop = image[ymin:ymax, xmin:xmax]

    return crop

def main():
    args = build_argparser().parse_args()

    init_models()

    feed = InputFeeder(args.input_type, args.input)
    feed.load_data()

    width, height = feed.get_input_shape()

    for frame in feed.next_batch():
        ### Extract face from frame
        if (frame is None):
            print("INFO: Empty frame found. Ending stream now.")
            break

        box_coords = face_detection.predict(frame)
        print(f"Boxes: {box_coords}")
        coords = box_coords[0]
        xmin = int(coords[0] * width)
        ymin = int(coords[1] * height)
        xmax = int(coords[2] * width)
        ymax = int(coords[3] * height)
        face = crop_rect(frame, (xmin, ymin, xmax, ymax))

        face_height, face_width, _ = face.shape
        print(f"Face shape: ({face_width}, {face_height})")

        ### Extract left and right eye from face
        eye_landmarks = facial_landmarks_detection.predict(face)
        landmarks_pos = [(int(l[0] * face_width), int(l[1] * face_height)) for l in eye_landmarks]
        left_eye_pos = landmarks_pos[0]
        right_eye_pos = landmarks_pos[1]
        left_eye_coords = [left_eye_pos[0] - x_offset, left_eye_pos[1] - y_offset, left_eye_pos[0] + x_offset, left_eye_pos[1] + y_offset]
        right_eye_coords = [right_eye_pos[0] - x_offset, right_eye_pos[1] - y_offset, right_eye_pos[0] + x_offset, right_eye_pos[1] + y_offset]
        # Zero out any negative values
        for i in range(4):
            left_eye_coords[i] = max(left_eye_coords[i], 0)
            right_eye_coords[i] = max(right_eye_coords[i], 0)
        print(f"Left eye pos: {left_eye_coords}")
        left_eye = crop_rect(face, left_eye_coords)
        right_eye = crop_rect(face, right_eye_coords)

        cv2.imshow("Test", frame)
        if cv2.waitKey(1) == 27:
            break

        ### Extract yaw, pitch and roll angles from face
        head_pose_angles = head_pose_estimation.predict(face)

        ### Extract gaze vector given both eyes and angles
        gaze_vector = gaze_estimation.predict(left_eye, right_eye, head_pose_angles)

        print(f"Gaze Vector: {gaze_vector}")
        print("\n")
    
    feed.close()

if __name__ == "__main__":
    main()