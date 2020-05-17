from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation

from input_feeder import InputFeeder
from mouse_controller import MouseController

import logging as log
from argparse import ArgumentParser

import cv2

### Model IR paths
path_face_detection = "models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml"
path_facial_landmarks_detection = "models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml"
path_head_pose_estimation = "models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml"
path_gaze_estimation = "models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml"

### Model global variables
face_detection = None
facial_landmarks_detection = None
head_pose_estimation = None
gaze_estimation = None

### Offsets to crop eyes from face
x_offset = 35
y_offset = 20

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-t", "--input_type", required=True, type=str, \
        help="Type of input. Valid values are 'image', 'cam', 'video'")
    
    parser.add_argument("-i", "--input", required=False, type=str, \
        help="Path to input file. Required if input type is 'image' or 'video'")

    parser.add_argument("-r", "--results", required=False, action="store_true", \
        help="Show intermediate model results")

    parser.add_argument("-d", "--device", required=False, type=str, default="CPU", \
        help="Set the device to run inference on (default cpu)")

    return parser

### Initiate & load all required models
def init_models(device="CPU"):
    # Using global variables, not defining new variables
    global face_detection
    global facial_landmarks_detection
    global head_pose_estimation
    global gaze_estimation

    log.info("Loading Face Detection model...")
    face_detection = Face_Detection(path_face_detection, device)
    face_detection.load_model()
    log.info("DONE\n")

    log.info("Loading Face Landmarks Detection model...")
    facial_landmarks_detection = Facial_Landmarks_Detection(path_facial_landmarks_detection, device)
    facial_landmarks_detection.load_model()
    log.info("DONE\n")

    log.info("Loading Head Pose Estimation model...")
    head_pose_estimation = Head_Pose_Estimation(path_head_pose_estimation, device)
    head_pose_estimation.load_model()
    log.info("DONE\n")

    log.info("Loading Gaze Estimation model...")
    gaze_estimation = Gaze_Estimation(path_gaze_estimation, device)
    gaze_estimation.load_model()
    log.info("DONE\n")

### Crop rectangle from given coordinates
def crop_rect(image, coords):
    xmin = coords[0]
    ymin = coords[1]
    xmax = coords[2]
    ymax = coords[3]

    crop = image[ymin:ymax, xmin:xmax]

    return crop

def main():
    # Get command line arguments
    args = build_argparser().parse_args()

    # Set log level to INFO
    log.basicConfig(level = log.INFO, format = '%(levelname)s: %(message)s')

    # Whether to show intermediate results from models
    show_results = args.results

    init_models(args.device)

    feed = InputFeeder(args.input_type, args.input)
    feed.load_data()
    width, height = feed.get_input_shape()

    controller = MouseController("medium", "fast")
    controller.move_to_center()

    for frame in feed.next_batch():
        ### Extract face from frame
        if (frame is None):
            log.info("Empty frame found. Ending stream now.")
            break

        box_coords = face_detection.predict(frame)
        # continue with next frame with no face is detected
        if (len(box_coords) == 0):
            continue

        face_coords = box_coords[0]
        xmin = int(face_coords[0] * width)
        ymin = int(face_coords[1] * height)
        xmax = int(face_coords[2] * width)
        ymax = int(face_coords[3] * height)
        face = crop_rect(frame, (xmin, ymin, xmax, ymax))
        face_height, face_width, _ = face.shape

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
        left_eye = crop_rect(face, left_eye_coords)
        right_eye = crop_rect(face, right_eye_coords)


        ### Extract yaw, pitch and roll angles from face
        head_pose_angles = head_pose_estimation.predict(face)

        ### Extract gaze vector given both eyes and angles
        gaze_vector = gaze_estimation.predict(left_eye, right_eye, head_pose_angles)

        if show_results:
            # Draw face box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255))

            # Draw eyes box
            coord_min_left_eye = (left_eye_coords[0] + xmin, left_eye_coords[1] + ymin)
            coord_max_left_eye = (left_eye_coords[2] + xmin, left_eye_coords[3] + ymin)
            coord_min_right_eye = (right_eye_coords[0] + xmin, right_eye_coords[1] + ymin)
            coord_max_right_eye = (right_eye_coords[2] + xmin, right_eye_coords[3] + ymin)
            cv2.rectangle(frame, coord_min_left_eye, coord_max_left_eye, (255, 0, 0))
            cv2.rectangle(frame, coord_min_right_eye, coord_max_right_eye, (255, 0, 0))

            # Draw gaze vector from each eye
            magnitude = 120
            pos_left_eye = (left_eye_pos[0] + xmin, left_eye_pos[1] + ymin)
            pos_right_eye = (right_eye_pos[0] + xmin, right_eye_pos[1] + ymin)
            coord_gaze_left_eye = (pos_left_eye[0] + int(gaze_vector[0] * magnitude), pos_left_eye[1] + int(gaze_vector[1] * magnitude) * -1)
            coord_gaze_right_eye = (pos_right_eye[0] + int(gaze_vector[0] * magnitude), pos_right_eye[1] + int(gaze_vector[1] * magnitude) * -1)
            cv2.arrowedLine(frame, pos_left_eye, coord_gaze_left_eye, (0, 0, 255), 2)
            cv2.arrowedLine(frame, pos_right_eye, coord_gaze_right_eye, (0, 0, 255), 2)

            log.info(f"Face box coords: ({xmin}, {ymin}), ({xmax}, {ymax})")
            log.info(f"Left Eye coords: {pos_left_eye}, Right Eye coords: {pos_right_eye}")
            log.info(f"Head pose angles: {head_pose_angles}")
            log.info(f"Gaze Vector: {gaze_vector}\n")

            cv2.imshow("Results", frame)
            # Stop if Esc key is pressed
            if cv2.waitKey(1) == 27:
                log.warning("Esc key pressed, inference interrupted!")
                break

        # controller.move(gaze_vector[0], gaze_vector[1])

    feed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()