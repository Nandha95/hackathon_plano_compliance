import tensorflow as tf
import numpy as np
import cv2

from app.config import SEGMENTATION_MODEL_PATH

class SegmentationModel:
    """
    This class loads the Tensorflow model, and does the following:
    1. Runs object detection on the image.
    2. Gets the bounding boxes of the detected objects
    """

    def __init__(self):

