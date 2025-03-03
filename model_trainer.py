
import os
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf
from hydra import compose, initialize

from model import build_DenseNet121
from utils.utils import setup_evnironment_vars
from utils.weighted_loss import get_weighted_loss

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
AUTOTUNE = tf.data.AUTOTUNE

# https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")

# ## Load Datasets
TRAIN_CSV = Path(cfg.DATASET_DIRS.TRAIN_CSV)
VALID_CSV = Path(cfg.DATASET_DIRS.VALID_CSV)

DENSENET_WEIGHT_PATH = cfg.PRETRAIN_MODEL.DENSENET_WEIGHT_PATH
datasets = 'datasets/sample'
dataset_path = Path(datasets)
TRAIN_IMG_DIR = dataset_path /'sample/images'
CSV_PATH = dataset_path /'sample/sample_labels.csv'
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
IMAGE_SIZE = cfg.TRAIN.IMG_SIZE
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE
NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS


def load_image(image_name, label):
    full_path = tf.strings.join([f'{TRAIN_IMG_DIR}/', image_name])
    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])  # Resize to the desired size
    label = tf.cast(label, tf.float32)
    return image, label

def load_image_valid(image_name, label):
    full_path = tf.strings.join(['datasets/images-small/', image_name])
    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])  # Resize to the desired size
    label = tf.cast(label, tf.float32)
    return image, label

def main()-> None:
    setup_evnironment_vars()


    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        raise Exception("No GPU found")
    else:
        print(f'[INFO] {found_gpu=}, {tf.__version__=}')

    sample_df = pd.read_csv(CSV_PATH)
    valid_csv= 'datasets/valid-small.csv'
    valid_df = pd.read_csv(f"{valid_csv}")
    valid_df.drop('PatientId', axis=1, inplace=True)

    # Data Clean up
    # Droping duplicates
    sample_df.drop_duplicates(subset=['Patient ID'], inplace=True)
    sample_labels = ['Image Index', 'Finding Labels']
    sample_df = sample_df[sample_labels]

    # sample_cat_df = sample_df['Finding Labels'].str.get_dummies(sep='|').astype('float32')

    # Prepare new csv file with useful information and format
    useful_data_df= sample_df[['Image Index', 'Finding Labels']]
    useful_data_df = useful_data_df.dropna()
    images_df = useful_data_df['Image Index'] 

    # One hot encoding
    # drop_colums = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    #        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    #        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    # 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    labels =['Atelectasis','Effusion','Infiltration', 'Mass','Nodule']

    # labels =['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    #     'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    #     'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


    train_cat_labels_df = useful_data_df['Finding Labels'].str.get_dummies(sep='|').astype('float32')
    train_cat_labels_df = train_cat_labels_df[labels]
    valid_images = valid_df.Image
    valid_labels_df = valid_df[labels]

    # Load Training Dataset from Dataframe
    train_ds = tf.data.Dataset.from_tensor_slices((images_df,  train_cat_labels_df.values))
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_images.values,  valid_labels_df.values))
    valid_ds = valid_ds.map(load_image_valid, num_parallel_calls=AUTOTUNE)

    # ## Image Processing

    # #### Normalization

    # Create normalization layer
    normalization_layer = tf.keras.layers.Normalization()
    # Compute the mean and variance using the training data
    # We need to convert the dataset to numpy to compute statistics
    images_for_stats = []
    for images, _ in train_ds.take(int(len(images_df)*0.25)): 
        images_for_stats.append(images)
    images_for_stats = tf.concat(images_for_stats, axis=0)
    normalization_layer.adapt(images_for_stats)


    # #### Augmentation 
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.10),                    # Small rotation
        tf.keras.layers.RandomTranslation(0.05, 0.05),           # Translation
        tf.keras.layers.RandomContrast(0.1),                     # Contrast adjustment
        tf.keras.layers.RandomBrightness(0.1),                   # Brightness adjustment
        tf.keras.layers.RandomZoom(0.1, 0.1)])                   # Random zoom


    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32) #/ 255.0
        # image = tf.keras.applications.densenet.preprocess_input(image)
        image = normalization_layer(image)
        return image, label


    train_ds = train_ds.map(lambda image, label: (data_augmentation(image), label) , num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(preprocess_image,num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).shuffle(len(images_df))
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    valid_ds = valid_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    # Class Imbalance Handling
    # Compute Class Frequencies

    N = train_cat_labels_df.shape[0]
    positive_frequencies = (train_cat_labels_df==1).sum()/N
    negative_frequencies = (train_cat_labels_df==0).sum()/N

    pos_weights = negative_frequencies.values
    neg_weights = positive_frequencies.values

    # ## Model Development
    # ### Load and Prepare DenseNet121 Model
    #'imagenet',

    model = build_DenseNet121(image_size=IMAGE_SIZE, num_classes=len(labels))

    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='AUC'), 
    ]

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE), 
                loss=get_weighted_loss(pos_weights, neg_weights),
            metrics=METRICS)     


    CHECK_POINT_DIR = 'exported_models'
    checkpoint_prefix = os.path.join(CHECK_POINT_DIR, "ckpt_{epoch}")
    LOG_DIR = 'logs'


    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True,
                                            save_best_only=True,
                                            monitor='val_loss',
                                            mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ]
    mlflow.set_experiment("/mlfow_chest_xray")

    mlflow.tensorflow.autolog()
    model.fit(train_ds, 
            validation_data=valid_ds,
            epochs = NUM_EPOCHS,
            callbacks=callbacks)


if __name__ == '__main__':
    main()


