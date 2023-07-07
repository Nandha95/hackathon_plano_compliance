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
from app.model.sku_classifier.SIFT import find_similar_images
from app.config import *
import pandas as pd

@st.cache_resource
def load_model():
    obj = DetectorModel()
    return obj

@st.cache_data
def detect_skus(_det_model: DetectorModel, det_image: np.ndarray, path):
    _det_model.add_image(det_image, DETECTION_TARGET_SIZE, output_path=str(path))
    _det_model.preprocess_image()
    _det_model.run_detection()
    _det_model.process_detection()
    _det_model.crop_images()
    _det_model.save_image()
    _det_model.save_detection_pickle()
    return _det_model.output_path


def draw_on_image(image, detections) -> np.ndarray:
    ori_image = image.copy()
    for i, detection in enumerate(detections):
        boxes = detection['coords']
        score = detection['score']
        # print(f"{i}:{coords}")
        name = ('NA' if detection['name']== 'object'  else detection['name'] )+ ' ' + '{p:.2f}'.format(p=float(detection['score']))
        if name == 'object':
            continue
        cv2.rectangle(ori_image, boxes[:2], boxes[2:], [random.randint(0, 255) for _ in range(3)], 2)
        cv2.putText(
            ori_image,
            name,
            (boxes[2], boxes[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            [225, 255, 255],
            thickness=1
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
    directory = Path(DETECTION_OUTPUT_PATH).joinpath(full_filename)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory

model = load_model()


st.title("Planogram Compliance")

with st.container():
    st.markdown("### Upload an image to detect")
    uploaded_images = st.file_uploader(
        label="Upload Image here",
        accept_multiple_files=True,
        type=['jpg','png']
    )


if not uploaded_images:
    st.error("Please upload images to proceed.")

image_with_bounding_box = []
image_filepaths = []
class_coords = []

if uploaded_images:
    for uploaded_image in uploaded_images:
        filepath = get_image_path(uploaded_image.name)
        image_filepaths.append(filepath)
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite(str(filepath.joinpath(uploaded_image.name)), opencv_image)
        # image = np.array(image)
        detection_output_path = detect_skus(model, opencv_image, path=filepath)
        print(f"DETECTION_OUTPUT_PATH: {detection_output_path}")
        image_array, image_paths = preprocess_image(path=detection_output_path)
        predictions = predict(image_array)
        label = [predictions[i].argmax() for i in range(predictions.shape[0])]
        prob = [predictions[i][label[i]] for i in range(predictions.shape[0])]
        image_coords = pickle.load(open(Path(detection_output_path).joinpath('detections.pickle'), 'rb'))
        for l, path, prob in zip(label, image_paths, prob):
            ind = int(path.split('_')[-1].split('.')[0])
            try:
                # print(path.split('_')[-1].split('.')[0],l,p)
                image_coords[ind]['name'] = str(l)
                image_coords[ind]['score'] = str(prob)
            except IndexError:
                print(l, path, prob, ind)
        pickle.dump(image_coords, open(Path(detection_output_path).joinpath('classification.pickle'), 'wb'))
        image_with_bounding_box.append(draw_on_image(cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR), image_coords))
        class_coords.append(image_coords)

st.write(image_filepaths)

if uploaded_images:
    with st.container():
        tab1, tab2 = st.tabs(['Images', 'Tables'])
        with tab1:
            st.write("Detection Images")
            st.image(image_with_bounding_box)
        with tab2:
            st.write("Detection Tables")
            for coord_set in class_coords:
                st.dataframe(pd.DataFrame(coord_set).reset_index()['name'].value_counts().reset_index())

