import sys
sys.path.append(".")  # nopep8
from app.controller.preprocess import PreprocessImage


def preprocess_image():
    obj = PreprocessImage()
    obj.create_dataframe_from_directory()
    # Preprocess the training data (with data augmentation)
    obj.preprocess_data(augmentation=True)


preprocess_image()
