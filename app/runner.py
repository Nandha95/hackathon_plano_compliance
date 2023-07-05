import sys
sys.path.append(".")  # nopep8
from app.controller.preprocess import PreprocessImage
from app.model.Segmentation.detector import DetectorModel
import cv2
from app.config import *
from app.model.sku_classifier.EffiecientNet import load_data,construct_model,train_model,predict
def preprocess_image(train=False):
    obj = PreprocessImage()
    if train:
        # obj.create_dataframe_from_directory()
        # Preprocess the training data (with data augmentation)
        # obj.preprocess_data(augmentation=True)
        X_train, X_test, y_train, y_test = load_data()
        model = construct_model()
        train_model(model, X_train,y_train,X_test, y_test)
    else:
        return obj.preprocess_test_data()
def run_detection():
    image = cv2.imread("app/data/input/train_shelf_images/train1.jpg")
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
    # 
    train = sys.argv[0] if sys.argv[1]=="1" else False
    if train:
        preprocess_image(train) 
    else:
        run_detection()
        x,image_paths = preprocess_image(train =False)
        predictions = predict(x)
        label = [predictions[i].argmax() for i in predictions.shape[0]]
        for l,p in zip(label, image_paths):
            print(l,p)