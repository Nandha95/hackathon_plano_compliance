SKU_CATLOG_PATH = "app/data/input/sku_catalog/"
# SKU_CATLOG_AUGMENTED_PATH = "app/data/input/sku_catalog_augment/"
SKU_CATLOG_AUGMENTED_PATH = "app/data/temp/"
TRAIN_SHELF_IMAGE_PATH = "app/data/input/train_shelf_images/"


SKU_CATLOG_CSV_PATH = "app/data/processed_data/sku_catlog.csv"

SKU_CATLOG_AUGMENTED_IMAGE_ARRAY_PATH = 'app/data/processed_data/sku_catalog_processed.npy'
SKU_CATLOG_AUGMENTED_LABEL_ARRAY_PATH = 'app/data/processed_data/sku_catalog_label_processed.npy'

# variables
IMAGE_PATH = "image_path"
LABEL = "class"
TARGET_SIZE = (224, 224)
AUGMENT_COUNT = 216

train_test_ratio = .75
SKU_CATLOG_SIZE = 39
checkpoint_path = "app/data/model/classifier/run3/cp-EffiecientNet-{epoch:04d}.ckpt"
model_path = "app/data/model/classifier/run3/"
# checkpoint_dir = os.path.dirname(checkpoint_path)

IMAGE_WITH_BOUNDINGBOX_PATH = "app/data/model/Segmentation_output/"
IMAGE_WITH_BOUNDINGBOX_COORDS_PATH = "app/data/model/Segmentation_output/labels"