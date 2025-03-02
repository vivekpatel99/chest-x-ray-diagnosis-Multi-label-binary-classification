import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)

found_gpu = tf.config.list_physical_devices('GPU')
if not found_gpu:
    raise Exception("No GPU found")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra import compose, initialize
from tensorflow.keras import backend as K
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
)
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

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.10),                    # Small rotation
    tf.keras.layers.RandomTranslation(0.05, 0.05),           # Translation
    tf.keras.layers.RandomContrast(0.1),                     # Contrast adjustment
    tf.keras.layers.RandomBrightness(0.1),                   # Brightness adjustment
    tf.keras.layers.RandomZoom(0.1, 0.1)])                   # Random zoom

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_df.Image.values,  train_df_labels.values))
train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
# train_ds = train_ds.map(lambda image, label: (data_augmentation(image), label) , num_parallel_calls=AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_df.Image,  valid_df_labels))
valid_ds = valid_ds.map(load_image, num_parallel_calls=AUTOTUNE)

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) 
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=len(train_df)).batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

valid_ds = valid_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

N = train_df_labels.shape[0]
positive_frequencies = (train_df_labels==1).sum()/N
negative_frequencies = (train_df_labels==0).sum()/N

pos_weights = negative_frequencies.values
neg_weights = positive_frequencies.values


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
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
# def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
#     """
#     Return weighted binary cross-entropy loss function with vectorized operations.
    
#     Args:
#       pos_weights (np.array): array of positive weights for each class, size (num_classes)
#       neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
#     Returns:
#       weighted_loss (function): weighted loss function
#     """
#     # Convert weights to tensors
#     pos_weights = tf.constant(pos_weights, dtype=tf.float32)
#     neg_weights = tf.constant(neg_weights, dtype=tf.float32)
    
#     def weighted_loss(y_true, y_pred):
#         """
#         Vectorized weighted binary cross-entropy loss.
        
#         Args:
#             y_true (Tensor): Tensor of true labels, size (batch_size, num_classes)
#             y_pred (Tensor): Tensor of predicted labels, size (batch_size, num_classes)
#         Returns:
#             loss (float): overall scalar loss 
#         """
#         # Cast inputs to float32
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
#         # Vectorized implementation (no loop)
#         # Create weight tensor based on y_true
#         weights = tf.where(
#             tf.equal(y_true, 1.0),
#             pos_weights,  # Use positive weights where y_true is 1
#             neg_weights   # Use negative weights where y_true is 0
#         )
        
#         # Standard binary cross-entropy formula
#         bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
#         # Apply weights and calculate mean
#         weighted_bce = weights * bce
        
#         # Average over all classes and examples
#         return tf.reduce_mean(weighted_bce)
    
#     return weighted_loss



base_model = DenseNet121(
     include_top=False,
     weights='imagenet',
     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)  
)
base_model.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output

x = GlobalAveragePooling2D()(x)

predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
# 2. Add proper adaptation layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Larger first layer
# x = tf.keras.layers.BatchNormalization()(x)            # Add batch normalization
# x = tf.keras.layers.Dropout(0.5)(x)                    # Increase dropout
# x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Add another layer
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.3)(x)
# predictions = Dense(len(labels), activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=predictions)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='AUC'), 
]

LEARNING_RATE =  0.0001
EPOCHS = 50
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
               loss= get_weighted_loss(pos_weights, neg_weights),
              metrics=METRICS)    

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
                    # steps_per_epoch=steps_per_epoch, 
                    # validation_steps=validation_steps, 
                    epochs = EPOCHS,
                    callbacks=callbacks)

