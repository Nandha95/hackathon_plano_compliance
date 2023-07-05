import onnxruntime as rt
import numpy as np
import cv2
from app.config import DETECTION_MODEL_PATH, DETECTION_NAMES, DETECTION_OUTPUT_PATH
from app.controller.preprocess import PreprocessImage
from typing import Tuple
from pathlib import Path
import pickle

class DetectorModel:
    """
    This class loads the Tensorflow model, and does the following:
    1. Runs object detection on the image.
    2. Gets the bounding boxes of the detected objects
    """
    session = rt.InferenceSession(DETECTION_MODEL_PATH)
    input_name = [i.name for i in session.get_inputs()]  # Name of the input node of the model.
    output_name = [i.name for i in session.get_outputs()]  # Name of the output node of the model.

    def __init__(
            self,
            image: np.ndarray,
            new_shape
    ) -> None:
        """
        :param image: The numpy array of the image on which object detection will be done. Needs to be a
        cv2 image, as cv2 operations will be performed on the image.
        """
        self.image = image
        self.new_shape = new_shape
        self.ratio = None  # The width:height ratio of the image after being scaled.
        self.dw_dh = None  # The new width and height of the image after being scaled.
        self.image_array = None
        self.input = None  # The input used by the model. This is not the same as input_name defined above
        self.output = None  # The output used by the model. This is not the same as output_name defined above
        self.detections = []
        self.cropped_images = []

    def preprocess_image(
            self
    ) -> None:
        """
        This method resizes and converts the image to the required shape to run the model.
        :return:
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = self.image.copy()
        image, self.ratio, self.dw_dh = PreprocessImage.resize_and_pad(image, new_shape=self.new_shape)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        self.image_array = image.astype(np.float32)
        self.image_array /= 255

    def run_detection(self) -> None:
        """
        Using the image array created above, this method runs the inference, and gets the detections for
        each model
        """
        self.input = {self.input_name[0]: self.image_array}
        self.output = self.session.run(self.output_name, self.input)[0]
        print(f"{len(self.output)} Detections found!")

    def process_detection(self) -> list[dict]:
        """
        Using the above inference output, this method will get the bounding boxes of each detected object, scaled back
        to the original image size
        :return: List of detection dictionaries, each having the
        """

        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(self.output):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(self.dw_dh * 2)
            box /= self.ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = DETECTION_NAMES[cls_id]
            self.detections.append({
                'num': i,
                'name': name,
                'score': score,
                'coords': box
            })
        return self.detections

    def crop_images(self) -> None:
        """
        Cropping the images based on the value
        """
        for i, detection in enumerate(self.detections):
            coords = detection['coords']
            print(f"{i}:{coords}")
            try:
                self.cropped_images.append(self.image[coords[1]:coords[3], coords[0]:coords[2]])
            except Exception as e:
                print(f"Unable to crop detection {i}, with boxes: {box}")
                print(e)

    def detection_array(self) -> np.ndarray:
        """
        returns the list of detections as a numpy array.
        :return:
        """
        return np.array(self.detections)

    def cropped_image_array(self) -> np.ndarray:
        """
        returns the list of cropped images as a numpy array.
        :return:
        """
        return np.array(self.cropped_images)

    def save_detection_pickle(self) -> None:
        ""
        with open(DETECTION_OUTPUT_PATH.joinpath('detections.pickle'), 'wb') as pickle_file:
            pickle.dump(self.detections, pickle_file)

    def save_image(self) -> None:
        """
        TODO: Method to save the cropped images after detection
        :return:
        """
        for i, image in enumerate(self.cropped_images):
            try:
                # cv2.imshow("test", image)
                # cv2.waitKey(0)
                # output_file_path = DETECTION_OUTPUT_PATH.joinpath(f"image_{i}.jpg")
                # # print(output_file_path)
                cv2.imwrite(str(DETECTION_OUTPUT_PATH.joinpath(f"image_{i}.jpg")), image)
            except Exception as e:
                print(f"Could not save image {i}: {e}")
