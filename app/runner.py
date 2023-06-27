import sys
sys.path.append(".")  # nopep8
from app.controller.preprocess import PreprocessImage
from app.model.Segmentation.detector import DetectorModel
import cv2
from app.config import *
def preprocess_image():
    obj = PreprocessImage()
    obj.create_dataframe_from_directory()
    # Preprocess the training data (with data augmentation)
    obj.preprocess_data(augmentation=True)

def run_detection():
    image = cv2.imread("data/input/train_shelf_images/train1.jpg")
    obj = DetectorModel(
        image,
        new_shape=DETECTION_TARGET_SIZE
    )
    obj.preprocess_image()
    obj.run_detection()
    obj.process_detection()
    obj.crop_images()
    detection_labels = obj.detection_array()
    detection_iamges = obj.cropped_image_array()
    obj.save_image()


if __name__=="__main__":
    # preprocess_image()
    run_detection()