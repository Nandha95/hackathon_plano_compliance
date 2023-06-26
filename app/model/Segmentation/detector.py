import onnxruntime as rt
import numpy as np
import cv2
from app.config import DETECTION_MODEL_PATH
from app.controller.preprocess import PreprocessImage

class DetectorModel:
    """
    This class loads the Tensorflow model, and does the following:
    1. Runs object detection on the image.
    2. Gets the bounding boxes of the detected objects
    """
    session = rt.InferenceSession(DETECTION_MODEL_PATH)
    input_name = [i.name for i in session.get_inputs()] # Name of the input node of the model.
    output_name = [i.name for i in session.get_outputs()] # Name of the output node of the model.

    def __init__(
            self,
            image: np.ndarray,
            new_shape: (int, int)
    ) -> None:
        """
        :param image: The numpy array of the image on which object detection will be done. Needs to be a
        cv2 image, as cv2 operations will be performed on the image.
        :param image_name: The name of the image file. This will be used for saving later on.
        """
        self.image = image
        self.new_shape = new_shape
        self.ratio = None
        self.dw_dh = None
        self.image_array = None

    def preprocess_image(
            self
    ) -> None:
        """
        This method resizes and converts the image to the required shape to run the model.
        :return:
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = self.image.copy()
        image, self.ratio, self.dw_dh = PreprocessImage.resize_and_pad(image, new_shape=self.new_shape)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        self.image_array = image.astype(np.float32)
        self.image_array /= 255
