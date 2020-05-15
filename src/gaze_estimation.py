import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class Gaze_Estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_xml, device='CPU'):
        ### Initialize any class variables desired
        self.core = None
        self.exec_network = None
        self.device = device
        
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        try:
            self.network = IENetwork(model_xml, model_bin)
        except Exception as e:
            raise ValueError("Failed to load model. Check path for suitable file.")

        self.input_blobs = [blob for blob in self.network.inputs]
        self.output_blob = next(iter(self.network.outputs))

    def load_model(self):
        ### Load the model 
        self.core = IECore()
        self.check_model()

        ### Return the loaded inference plugin 
        self.exec_network = self.core.load_network(self.network, self.device)

    def predict(self, image_left_eye, image_right_eye, head_pose_angles):
        p_left_eye, p_right_eye, p_head_pose_angles = self.preprocess_input(image_left_eye, image_right_eye, head_pose_angles)
        self.exec_network.start_async(0, {self.input_blobs[0]: p_head_pose_angles, self.input_blobs[1]: p_left_eye, \
            self.input_blobs[2]: p_right_eye})
        status = self.exec_network.requests[0].wait(-1)
        gaze_vector = []
        if status == 0:
            outputs = self.exec_network.requests[0].outputs
            gaze_vector = self.preprocess_output(outputs)
        return gaze_vector

    def check_model(self):
        ### Check for supported layers
        supported_layers = self.core.query_network(self.network, self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if (len(unsupported_layers) > 0):
            print("ERROR: Unsupported layers found!")
            exit(1)

    def preprocess_input(self, image_left_eye, image_right_eye, head_pose_angles):
        image_input_shape = self.network.inputs[self.input_blobs[1]].shape
        angles_input_shape = self.network.inputs[self.input_blobs[0]].shape
        
        p_left_eye = cv2.resize(image_left_eye, (image_input_shape[3], image_input_shape[2]))
        p_left_eye = p_left_eye.transpose((2, 0, 1))
        p_left_eye = p_left_eye.reshape(1, *p_left_eye.shape)

        p_right_eye = cv2.resize(image_right_eye, (image_input_shape[3], image_input_shape[2]))
        p_right_eye = p_right_eye.transpose((2, 0, 1))
        p_right_eye = p_right_eye.reshape(1, *p_right_eye.shape)

        p_head_pose_angles = np.array(head_pose_angles).reshape(*angles_input_shape)

        return p_left_eye, p_right_eye, p_head_pose_angles

    def preprocess_output(self, outputs):
        gaze_vector = outputs[self.output_blob].flatten()
        return gaze_vector
