# required libraries
import os
import pandas as pd
from PIL import Image
import numpy as np
from app.config import *
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from typing import Tuple

class PreprocessImage:
    def create_dataframe_from_directory(self):
        image_paths = []
        labels = []

        for filename in os.listdir(SKU_CATLOG_PATH):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
                image_path = os.path.join(SKU_CATLOG_PATH, filename)
                image_paths.append(image_path)
                # Extract the label from the image file name
                label = os.path.splitext(filename)[0]
                labels.append(label)

        self.df = pd.DataFrame({IMAGE_PATH: image_paths, LABEL: labels})
        self.df.to_csv(SKU_CATLOG_CSV_PATH, index=False)

    def normalize_image(self, image, label=None, aug_cnt=None):

        image = image / 255.0  # Normalize pixel values
        # if label == '0':
        #     plt.imshow(image)
        #     plt.savefig(SKU_CATLOG_AUGMENTED_PATH +
        #                 'image_{}_{}.jpg'.format(label, aug_cnt))
        #     image = tf.expand_dims(image, axis=0)  # Adzd batch dimension

        return image

    def preprocess_data(self, augmentation=False):
        preprocessed_images = []
        labels = []

        for index, row in tqdm(self.df.iterrows()):
            image_path = row[IMAGE_PATH]
            image_class = row[LABEL]
            image = load_img(
                image_path, target_size=TARGET_SIZE)
            image_array = img_to_array(image)
            if augmentation:
                # Data augmentation for training data
                data_generator = ImageDataGenerator(rotation_range=.2, width_shift_range=0.1, height_shift_range=0.1,
                                                    shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
                image_array = image_array.reshape((1,) + image_array.shape)
                augmented_images = data_generator.flow(
                    image_array, batch_size=1)

                # Generate 5 augmented samples per original image
                for i in range(AUGMENT_COUNT):
                    augmented_image = augmented_images.next()
                    preprocessed_image = self.normalize_image(
                        augmented_image[0], image_class, i)
                    preprocessed_images.append(preprocessed_image)
                    labels.append(image_class)

        preprocessed_images = np.array(preprocessed_images)
        labels = np.array(labels)
        print(preprocessed_images.shape, labels.shape)
        np.save(SKU_CATLOG_AUGMENTED_IMAGE_ARRAY_PATH, preprocessed_images)
        np.save(SKU_CATLOG_AUGMENTED_LABEL_ARRAY_PATH, tf.keras.utils.to_categorical(labels))
        print(np.unique(labels))
    def preprocess_test_data(self):
        preprocessed_images = []
        image_paths = []
        for filename in os.listdir(DETECTION_OUTPUT_PATH):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
                image_path = os.path.join(DETECTION_OUTPUT_PATH, filename)
                image = load_img(
                image_path, target_size=TARGET_SIZE)
                image_array = img_to_array(image)
                preprocessed_image = self.normalize_image(
                    image_array)
                preprocessed_images.append(preprocessed_image)
                image_paths.append(image_path)
        return np.array(preprocessed_images),image_paths
    @staticmethod
    def resize_and_pad(
            image: np.ndarray,
            new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        border_color = (114, 114, 114)
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color
        )  # add border
        return image, r, (dw, dh)
