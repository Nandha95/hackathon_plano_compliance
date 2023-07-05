import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from app.model.Segmentation.detector import DetectorModel
from app.config import DETECTION_MODEL_PATH, DETECTION_TARGET_SIZE

@st.cache_resource
def load_model():
    obj = DetectorModel()
    obj.load_model(DETECTION_MODEL_PATH)
    return obj

@st.cache_data
def detect_skus(_det_model: DetectorModel, image: np.ndarray):
    _det_model.add_image(image, DETECTION_TARGET_SIZE)
    _det_model.preprocess_image()
    _det_model.run_detection()
    _det_model.process_detection()
    _det_model.crop_images()
    detection_labels = _det_model.detection_array()
    detection_images = _det_model.cropped_image_array()
    image_with_labels = _det_model.image_with_bounding_boxes()
    return detection_images, detection_labels, image_with_labels


model = load_model()


st.title("Planogram Compliance")

with st.container():
    st.markdown("### Upload an image to detect")
    uploaded_images = st.file_uploader(
        label="Upload Image(s) here",
        accept_multiple_files=True,
        type=['jpg','png']
    )

image_with_bounding_box = []

if uploaded_images:
    for uploaded_image in uploaded_images:
        image = np.array(Image.open(uploaded_image))
        detection_images, detection_labels, image_with_labels = detect_skus(model, image)
        image_with_bounding_box.append(image_with_labels)


with st.container():
    st.image(image_with_bounding_box)