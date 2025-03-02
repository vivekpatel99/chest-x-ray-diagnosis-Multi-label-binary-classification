
import os

import tensorflow as tf
from sklearn import base

tf.random.set_seed(42)


found_gpu = tf.config.list_physical_devices('GPU')
if not found_gpu:
    raise Exception("No GPU found")
found_gpu, tf.__version__


import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra import compose, initialize
from tensorflow.keras import backend as K
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
    print(cfg.DATASET_DIRS.IMAGES_DIR)


IMAGE_DIR = Path(cfg.DATASET_DIRS.IMAGES_DIR)
TRAIN_CSV = Path(cfg.DATASET_DIRS.TRAIN_CSV)
VALID_CSV = Path(cfg.DATASET_DIRS.VALID_CSV)
TEST_CSV = Path(cfg.DATASET_DIRS.TEST_CSV)

DENSENET_WEIGHT_PATH = cfg.PRETRAIN_MODEL.DENSENET_WEIGHT_PATH

BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
IMAGE_SIZE = cfg.TRAIN.IMG_SIZE

train_df = pd.read_csv(f"{TRAIN_CSV}")
valid_df = pd.read_csv(f"{VALID_CSV}")

test_df = pd.read_csv(f"{TEST_CSV}")

column_names = list(train_df.columns)
labels = column_names[1:]
labels.remove('PatientId')
train_df_labels = train_df[labels]
valid_df_labels = valid_df[labels]
test_df_labels = test_df[labels]

def check_for_dataleakage(df1, df2, patentId_col='PatientId'):

    df1_patients_unique = df1[patentId_col].unique()
    df2_patients_unique = df2[patentId_col].unique()

    patients_in_common = np.intersect1d(df1_patients_unique, df2_patients_unique)

    leakage = True if (len(patients_in_common) > 0) else False
    return leakage


def is_image_exists(df):

    temp_df = df.copy() 
    for idx, image_path in enumerate(df.Image.values):
        
        if not Path(f"{IMAGE_DIR/image_path}").exists():
            print(f"Image {IMAGE_DIR/image_path} does not exist :{idx=}")
            temp_df = temp_df.drop(idx)

    return temp_df
def load_image(image_name, label):
    full_path = tf.strings.join([f'{IMAGE_DIR}/', image_name])
    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])  
    label = tf.cast(label, tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((train_df.Image.values,  train_df_labels.values))
train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((valid_df.Image.values,  valid_df_labels.values))
valid_ds = valid_ds.map(load_image, num_parallel_calls=AUTOTUNE)


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    mean, varience = tf.nn.moments(image, axes=[0, 1, 2])
    image = (image - mean) / tf.math.sqrt(varience + 1e-7)
    return image, label

train_ds = train_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).shuffle(BATCH_SIZE*4)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

valid_ds = valid_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

N = train_df_labels.shape[0]
positive_frequencies = (train_df_labels==1).sum()/N
negative_frequencies = (train_df_labels==0).sum()/N


pos_weights = negative_frequencies
neg_weights = positive_frequencies

pos_contirbution = positive_frequencies * pos_weights
neg_contribution = negative_frequencies * neg_weights

weighted_data_df = pd.DataFrame(list(pos_contirbution.items()), columns=['class', 'positives'])
weighted_data_df['negatives'] = neg_contribution.values

weighted_data_df.plot.bar(x='class')

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0

        for i in range(len(pos_weights)):
            y = y_true[:, i]
            f_of_x = y_pred[:, i]

            f_of_x_log = K.log(f_of_x + epsilon)
            f_of_x_1_min_log = K.log((1-f_of_x) + epsilon)

            first_term = pos_weights[i] * y * f_of_x_log
            sec_term = neg_weights[i] * (1-y) * f_of_x_1_min_log
            loss_per_col = - K.mean(first_term + sec_term)
            loss += loss_per_col
        return loss

    return weighted_loss

LEARNING_RATE = 0.001

base_model = DenseNet121(
     include_top=False,
     weights=DENSENET_WEIGHT_PATH, 
     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)  
)
base_model.trainable = False
x = base_model.output


x = GlobalAveragePooling2D()(x)


predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
               loss=get_weighted_loss(pos_weights, neg_weights),
              metrics=['accuracy'])     


length_of_training_dataset = len(train_df)
length_of_validation_dataset = len(valid_df)
steps_per_epoch = math.ceil(length_of_training_dataset/BATCH_SIZE)
validation_steps = math.ceil(length_of_validation_dataset/BATCH_SIZE)

CHECK_POINT_DIR = cfg.OUTPUTS.CHECKPOINT_PATH
checkpoint_prefix = os.path.join(CHECK_POINT_DIR, "ckpt_{epoch}")
LOG_DIR = cfg.OUTPUTS.LOG_DIR

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True,
                                        save_best_only=True,
                                        monitor='val_loss',
                                        mode='min'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
]

history = model.fit(train_ds, 
                    validation_data=valid_ds,
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps, 
                    epochs = 100,
                    callbacks=callbacks)

