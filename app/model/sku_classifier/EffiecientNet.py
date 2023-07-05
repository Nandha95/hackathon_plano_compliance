import efficientnet.keras as efn
import numpy as np
import sys

sys.path.append(".")
from app.config import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


def load_data():
    X = np.load(SKU_CATLOG_AUGMENTED_IMAGE_ARRAY_PATH)
    y = np.load(SKU_CATLOG_AUGMENTED_LABEL_ARRAY_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_ratio, random_state=1,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def construct_model():
    base_model = efn.EfficientNetB0(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), include_top=False,
                                    weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    predictions = Dense(SKU_CATLOG_SIZE, activation="softmax")(x)

    model_final = Model(inputs=base_model.input, outputs=predictions)

    model_final.compile(optimizers.legacy.RMSprop(learning_rate=0.0001, epsilon=1e-6),
                        loss='categorical_crossentropy', metrics=['categorical_accuracy', 'Precision', 'Recall'])
    return model_final


def train_model(model_final, X_train, y_train, X_test, y_test):
    model_final.load_weights("app/data/model/classifier/run_4/cp-EffiecientNet-0006.ckpt")

    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_best_only=True)
    eff_history = model_final.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), steps_per_epoch=30,
                                  batch_size=64, callbacks=[cp_callback], epochs=20)
    # import pdb
    # pdb.set_trace()
    # model_final.export(model_final, model_path+'EffiecientNet')


def predict(x):
    model = construct_model()
    # import pdb
    # pdb.se
    model.load_weights(model_path + "/model_weights.h5")
    return model.predict(x)

# predict(x)