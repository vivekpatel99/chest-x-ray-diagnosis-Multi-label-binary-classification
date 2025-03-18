
""""
# # NIH Chest X-ray Multi label Binary classification using Tensorflow Densenet121 (Transfer learning)
    # drop_colums = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    #        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    #        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    # 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',

    # labels =['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    #        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    #        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
"""
# ## Imports
import os

# Set environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from arrow import get

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)




import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import opendatasets as od
import pandas as pd
import seaborn as sns
from hydra import compose, initialize
from tensorflow.keras import backend as K

from src.data_loader.chest_x_ray_preprocessor import ChestXRayPreprocessor
from src.model.model import build_DenseNet121
from src.utils.logs import get_logger
from src.weighted_loss.weighted_loss import get_weighted_loss

# https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
    print(cfg.DATASET_DIRS.TRAIN_IMAGES_DIR)


# ## Constants
IMAGE_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE
CHECK_POINT_DIR = 'exported_models'
LOG_DIR = 'logs'

def main():
    log = get_logger(__name__, log_level=logging.INFO)
    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.info("No GPU found")
        raise Exception("No GPU found")
    
    # Look into the data directory
    datasets = 'datasets/sample'
    dataset_path = Path(datasets)
    dataset_path.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_dir():
        log.info("Downloading the dataset")
        dataset_url = 'https://www.kaggle.com/datasets/nih-chest-xrays/sample'
        od.download(dataset_url)

    CLASSES_NAME = ['Atelectasis','Effusion','Infiltration', 'Mass']#,'Nodule']

    preprocessor = ChestXRayPreprocessor(cfg, labels=CLASSES_NAME)
    train_ds, valid_ds, pos_weights, neg_weights = preprocessor.get_training_and_validation_datasets()

    to_monitor = 'val_AUC'
    mode = 'max'
    mlflow.tensorflow.autolog(log_models=True, 
                            log_datasets=False, 
                            log_input_examples=True,
                            log_model_signatures=True,
                            keras_model_kwargs={"save_format": "keras"},
                            checkpoint_monitor=to_monitor, 
                            checkpoint_mode=mode)

    model = build_DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), num_classes=len(CLASSES_NAME))
    log.info(f"Model summary: {model.summary()}")

    METRICS = [
        'binary_accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='AUC'), 
    ]

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), 
                loss=get_weighted_loss(pos_weights, neg_weights),
            metrics=METRICS)     


    callbacks = get_callbacks(to_monitor, mode)


    model.fit(train_ds, 
            validation_data=valid_ds,
            batch_size=BATCH_SIZE,
            epochs = NUM_EPOCHS,
            callbacks=callbacks)

    preprocessor = ChestXRayPreprocessor(cfg, labels=CLASSES_NAME)
    test_ds = preprocessor.get_test_dataset()
    result = model.evaluate(test_ds, return_dict=True)
    mlflow.log_metrics(result)





def get_callbacks(to_monitor, mode):
    checkpoint_prefix = os.path.join(CHECK_POINT_DIR, "ckpt_{epoch}.keras")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                            save_best_only=True, # Save only the best model based on val_loss
                                            monitor=to_monitor,
                                            mode=mode,
                                            verbose=1),  # Display checkpoint saving messages
        tf.keras.callbacks.ReduceLROnPlateau(monitor=to_monitor,
                                            mode=mode, factor=0.1, patience=5, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor=to_monitor,
                                            mode=mode, patience=10, restore_best_weights=True),
    ]
    
    return callbacks

if __name__ == '__main__':
    main()





