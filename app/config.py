from pathlib import Path
BASE_PATH = Path.cwd()


SKU_CATLOG_PATH = "app/data/input/sku_catalog/"
# SKU_CATLOG_AUGMENTED_PATH = "app/data/input/sku_catalog_augment/"
SKU_CATLOG_AUGMENTED_PATH = "app/data/temp/"
TRAIN_SHELF_IMAGE_PATH = "app/data/input/train_shelf_images/"


SKU_CATLOG_CSV_PATH = "app/data/processed_data/sku_catlog.csv"

SKU_CATLOG_AUGMENTED_IMAGE_ARRAY_PATH = 'app/data/processed_data/sku_catalog_processed.npy'
SKU_CATLOG_AUGMENTED_LABEL_ARRAY_PATH = 'app/data/processed_data/sku_catalog_label_processed.npy'

# OBJECT DETECTION CONFIGS
DETECTION_MODEL_PATH = BASE_PATH.joinpath('data/models/yolov7/yolov7-tiny.onnx')
print(BASE_PATH)
print(DETECTION_MODEL_PATH)
DETECTION_TARGET_SIZE = (640, 640)
DETECTION_NAMES = ['object']
DETECTION_OUTPUT_PATH = BASE_PATH.joinpath('data/processed_data/detections')


# variables
IMAGE_PATH = "image_path"
LABEL = "class"
TARGET_SIZE = (224, 224)
AUGMENT_COUNT = 50
