import cv2
from face_detection import Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation

import time

prec_face_detection = "FP16-INT8"
prec_landmarks_detection = "FP16-INT8"
prec_head_pose_estimation = "FP16-INT8"
prec_gaze_estimation = "FP16-INT8"

path_input = "media/demo.mp4"

path_face_detection = "models/intel/face-detection-adas-0001/" + prec_face_detection + "/face-detection-adas-0001.xml"
path_facial_landmarks_detection = "models/intel/landmarks-regression-retail-0009/" + prec_landmarks_detection + "/landmarks-regression-retail-0009.xml"
path_head_pose_estimation = "models/intel/head-pose-estimation-adas-0001/" + prec_head_pose_estimation + "/head-pose-estimation-adas-0001.xml"
path_gaze_estimation = "models/intel/gaze-estimation-adas-0002/" + prec_gaze_estimation + "/gaze-estimation-adas-0002.xml"

### Model global variables
face_detection = None
facial_landmarks_detection = None
head_pose_estimation = None
gaze_estimation = None

### Initiate & load all required models
def init_models(device="CPU"):
    # Using global variables, not defining new variables
    global face_detection
    global facial_landmarks_detection
    global head_pose_estimation
    global gaze_estimation

    start = time.time()
    face_detection = Face_Detection(path_face_detection, device)
    face_detection.load_model()
    fd_load_time = (time.time() - start)

    start = time.time()
    facial_landmarks_detection = Facial_Landmarks_Detection(path_facial_landmarks_detection, device)
    facial_landmarks_detection.load_model()
    fld_load_time = (time.time() - start)

    start = time.time()
    head_pose_estimation = Head_Pose_Estimation(path_head_pose_estimation, device)
    head_pose_estimation.load_model()
    hpe_load_time = (time.time() - start)

    start = time.time()
    gaze_estimation = Gaze_Estimation(path_gaze_estimation, device)
    gaze_estimation.load_model()
    ge_load_time = (time.time() - start)

    return (fd_load_time, fld_load_time, hpe_load_time, ge_load_time)

### Crop rectangle from given coordinates
def crop_rect(image, coords):
    xmin = coords[0]
    ymin = coords[1]
    xmax = coords[2]
    ymax = coords[3]

    crop = image[ymin:ymax, xmin:xmax]

    return crop

def get_millis(seconds):
    return int(round(seconds * 1000))

def run_face_benchmark():
    fd_load_time, fld_load_time, hpe_load_time, ge_load_time = init_models()

    cap = cv2.VideoCapture(path_input)
    cap.open(path_input)

    width = int(cap.get(3))
    height = int(cap.get(4))

    fd_total_time = 0
    fld_total_time = 0
    hpe_total_time = 0
    ge_total_time = 0

    counter = 0
    while(cap.isOpened()):
        flag, frame = cap.read()
        if not flag:
            break

        counter += 1
        
        fd_start = time.time()
        box_coords = face_detection.predict(frame)
        fd_total_time += (time.time() - fd_start)

        face_coords = box_coords[0]
        xmin = int(face_coords[0] * width)
        ymin = int(face_coords[1] * height)
        xmax = int(face_coords[2] * width)
        ymax = int(face_coords[3] * height)
        face = crop_rect(frame, (xmin, ymin, xmax, ymax))
        face_height, face_width, _ = face.shape

        fld_start = time.time()
        eye_landmarks = facial_landmarks_detection.predict(face)
        fld_total_time += (time.time() - fld_start)

        landmarks_pos = [(int(l[0] * face_width), int(l[1] * face_height)) for l in eye_landmarks]
        left_eye_pos = landmarks_pos[0]
        right_eye_pos = landmarks_pos[1]
        left_eye_coords = [left_eye_pos[0] - 35, left_eye_pos[1] - 20, left_eye_pos[0] + 35, left_eye_pos[1] + 20]
        right_eye_coords = [right_eye_pos[0] - 35, right_eye_pos[1] - 20, right_eye_pos[0] + 35, right_eye_pos[1] + 20]
        
        # Zero out any negative values
        for i in range(4):
            left_eye_coords[i] = max(left_eye_coords[i], 0)
            right_eye_coords[i] = max(right_eye_coords[i], 0)
        left_eye = crop_rect(face, left_eye_coords)
        right_eye = crop_rect(face, right_eye_coords)

        hpe_start = time.time()
        head_pose_angles = head_pose_estimation.predict(face)
        hpe_total_time += (time.time() - hpe_start)

        ge_start = time.time()
        gaze_vector = gaze_estimation.predict(left_eye, right_eye, head_pose_angles)
        ge_total_time += (time.time() - ge_start)

    print("\n")
    print(f"Total frames in input: {counter}")
    print("=========== BENCHMARK ============")

    print(f"Face Detection ({prec_face_detection})")
    print(f"Load Time: {get_millis(fd_load_time)} ms   Total time: {get_millis(fd_total_time)} ms   fps: {round(counter / fd_total_time, 2)} frames/s")
    print("\n")

    print(f"Face Landmarks Detection ({prec_landmarks_detection})")
    print(f"Load Time: {get_millis(fld_load_time)} ms   Total time: {get_millis(fld_total_time)} ms   fps: {round(counter / fld_total_time, 2)} frames/s")
    print("\n")

    print(f"Head Pose Estimation ({prec_head_pose_estimation})")
    print(f"Load Time: {get_millis(hpe_load_time)} ms   Total time: {get_millis(hpe_total_time)} ms   fps: {round(counter / hpe_total_time, 2)} frames/s")
    print("\n")

    print(f"Gaze Estimation ({prec_gaze_estimation})")
    print(f"Load Time: {get_millis(ge_load_time)} ms   Total time: {get_millis(ge_total_time)} ms   fps: {round(counter / ge_total_time, 2)} frames/s")
    print("\n")

def main():
    run_face_benchmark()

if __name__ == "__main__":
    main()