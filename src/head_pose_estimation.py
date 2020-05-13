import os
import cv2
from openvino.inference_engine import IENetwork, IECore

class Head_Pose_Estimation:
    '''
    Class for the Head Pose Estimation Model.
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

        self.input_blob = next(iter(self.network.inputs))
        self.output_blobs = [blob for blob in self.network.outputs]

    def load_model(self):
        ### Load the model 
        self.core = IECore()

        ### Check for supported layers
        supported_layers = self.core.query_network(self.network, self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if (len(unsupported_layers) > 0):
            print("ERROR: Unsupported layers found!")
            exit(1)

        ### Return the loaded inference plugin 
        self.exec_network = self.core.load_network(self.network, self.device)

        return self.core

    def predict(self, image):
        p_image = self.preprocess_input(image)
        self.exec_network.start_async(0, {self.input_blob: p_image})
        status = self.exec_network.requests[0].wait(-1)
        angles = []
        if status == 0:
            outputs = self.exec_network.requests[0].outputs
            angles = self.preprocess_output(outputs)
        return angles

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        net_input_shape = self.network.inputs[self.input_blob].shape
        p_image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_image = p_image.transpose((2, 0, 1))
        p_image = p_image.reshape(1, *p_image.shape)
        return p_image

    def preprocess_output(self, outputs):
        angles = []
        for blob in self.output_blobs:
            angle = outputs[blob].item()
            angles.append(angle)
        return angles
