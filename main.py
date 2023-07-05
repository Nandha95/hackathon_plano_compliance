import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import random
import pickle
import time
from app.model.Segmentation.detector import DetectorModel
from app.controller.preprocess import PreprocessImage
from app.model.sku_classifier.EffiecientNet import predict
from app.config import *

@st.cache_resource
def load_model():
    obj = DetectorModel()
    obj.load_model(DETECTION_MODEL_PATH)
    return obj

@st.cache_data
def detect_skus(_det_model: DetectorModel, det_image: np.ndarray, path):
    _det_model.add_image(det_image, DETECTION_TARGET_SIZE, output_path=path)
    _det_model.preprocess_image()
    _det_model.run_detection()
    _det_model.process_detection()
    _det_model.crop_images()
    _det_model.save_image()
    _det_model.save_detection_pickle()
    return _det_model.detection_output_path


def draw_on_image(image, detections) -> np.ndarray:
    ori_image = image.copy()
    for i, detection in enumerate(detections):
        boxes = detections['coords']
        score = detection['score']
        # print(f"{i}:{coords}")
        name = detection['name'] + ' ' + str(detection['score'])
        cv2.rectangle(ori_image, boxes[:2], boxes[2:], [random.randint(0, 255) for _ in range(3)], 2)
        cv2.putText(
            ori_image,
            name,
            (boxes[0], boxes[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            thickness=2
        )
    return ori_image


@st.cache_data
def preprocess_image(path):
    obj = PreprocessImage()
    return obj.preprocess_test_data(PATH=path)

@st.cache_data
def classify_skus():
    pass


def get_image_path(filename):
    full_filename = uploaded_image.name+'_'+str(time.time())
    directory = BASE_OUTPUT_PATH.joinpath(full_filename)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory

model = load_model()


st.title("Planogram Compliance")

with st.container():
    st.markdown("### Upload an image to detect")
    uploaded_images = st.file_uploader(
        label="Upload Image(s) here",
        accept_multiple_files=True,
        type=['jpg','png']
    )


if not uploaded_images:
    st.error("Please upload images to proceed.")

image_with_bounding_box = []

image_filepaths = []

if uploaded_images:
    for uploaded_image in uploaded_images:
        filepath = get_image_path(uploaded_image.name)
        image_filepaths.append(filepath)
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite(str(filepath.joinpath(uploaded_image.name)), opencv_image)
        # image = np.array(image)
        detection_output_path = detect_skus(model, opencv_image, path=filepath)
        image_array, image_paths = preprocess_image(path=detection_output_path)
        predictions = predict(image_array)
        label = [predictions[i].argmax() for i in range(predictions.shape[0])]
        prob = [predictions[i][label[i]] for i in range(predictions.shape[0])]
        image_coords = pickle.load(open(detection_output_path.joinpath('detections.pickle'), 'rb'))
        for l, path, prob in zip(label, image_paths, prob):
            # print(path.split('_')[-1].split('.')[0],l,p)
            ind = int(path.split('_')[-1].split('.')[0])
            image_coords[ind]['name'] = str(l)
            image_coords[ind]['score'] = str(prob)
        image_with_bounding_box.append(draw_on_image(opencv_image, image_coords))

st.write(image_filepaths)


with st.container():
    st.image(image_with_bounding_box)