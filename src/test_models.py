import cv2
from face_detection import Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation

def test_face_detection():
    model = Face_Detection("models/intel/face-detection-adas-binary-0001/FP32/face-detection-adas-binary-0001.xml")
    model.load_model()
    image = cv2.imread("bin/sample.png")
    height, width, _ = image.shape
    box_coords = model.predict(image)
    count = 0
    face = None
    for box in box_coords:
        count += 1
        xmin = int(box[0] * width)
        ymin = int(box[1] * height)
        xmax = int(box[2] * width)
        ymax = int(box[3] * height)
        face = image[ymin:ymax, xmin:xmax]
        cv2.imwrite("bin/face" + str(count) + ".jpg", face)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    
    cv2.imshow("Result", image)
    cv2.waitKey()

def test_head_pose_estimation():
    model = Head_Pose_Estimation("models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml")
    model.load_model()
    image = cv2.imread("bin/face1.jpg")
    angles = model.predict(image)
    print("Angles: " + str(angles))

    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.5
    thickness = 1
    cv2.putText(image, "yaw:" + str(angles[0]), (5, 20), font, scale, color, thickness)
    cv2.putText(image, "pitch:" + str(angles[1]), (5, 40), font, scale, color, thickness)
    cv2.putText(image, "roll:" + str(angles[2]), (5, 60), font, scale, color, thickness)
    
    cv2.imshow("Result", image)
    cv2.waitKey()

def test_facial_landmarks_detection():
    model = Facial_Landmarks_Detection("models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml")
    model.load_model()
    image = cv2.imread("bin/face1.jpg")
    height, width, _ = image.shape
    eye_landmarks = model.predict(image)
    landmarks_pos = [(int(l[0] * width), int(l[1] * height)) for l in eye_landmarks]
    left_eye_coord = landmarks_pos[0]
    right_eye_coord = landmarks_pos[1]
    
    x_offset = 50
    y_offset = 25
    left_eye = image[left_eye_coord[1] - y_offset:left_eye_coord[1] + y_offset, left_eye_coord[0] - x_offset:left_eye_coord[0] + x_offset]
    right_eye = image[right_eye_coord[1] - y_offset:right_eye_coord[1] + y_offset, right_eye_coord[0] - x_offset:right_eye_coord[0] + x_offset]
    
    cv2.imwrite("bin/left_eye.jpg", left_eye)
    cv2.imwrite("bin/right_eye.jpg", right_eye)

    radius = 5
    color = (255, 0, 0)
    thickness = 5
    cv2.circle(image, left_eye_coord, radius, color, thickness)
    cv2.circle(image, right_eye_coord, radius, color, thickness)
    cv2.imshow("Result", image)
    cv2.waitKey()

def test_gaze_estimation():
    model = Gaze_Estimation("models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml")
    model.load_model()
    left_eye = cv2.imread("bin/left_eye.jpg")
    right_eye = cv2.imread("bin/right_eye.jpg")
    angles = [-14.323277473449707, -2.0438201427459717, 4.142961502075195]
    gaze_vector = model.predict(left_eye, right_eye, angles)
    print("GazeVector: " + str(gaze_vector))
    
def main():
    # test_face_detection()
    # test_head_pose_estimation()
    # test_facial_landmarks_detection()
    # test_gaze_estimation()

if __name__ == "__main__":
    main()