
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
import logging

# ## Imports
import os
from pathlib import Path

import mlflow
import numpy as np
import opendatasets as od
import seaborn as sns
from hydra import compose, initialize
from matplotlib import pyplot as plt
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import mixed_precision

from src.data_loader.chest_x_ray_preprocessor import ChestXRayPreprocessor
from src.model.model import build_DenseNet121
from src.utils.logs import get_logger
from src.utils.utils import plot_auc_curve
from src.weighted_loss.weighted_loss import get_weighted_loss

mixed_precision.set_global_policy('mixed_float16')
# Set environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)


# https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
    print(cfg.DATASET_DIRS.TRAIN_IMAGES_DIR)


# ## Constants
IMAGE_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE
OUPUT_DIR = Path(cfg.OUTPUTS.OUPUT_DIR)
OUPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECK_POINT_DIR = Path(cfg.OUTPUTS.CHECKPOINT_PATH)
CHECK_POINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = 'logs'

def main() -> None:
    log = get_logger(__name__, log_level=logging.INFO)
    
    mlflow.set_experiment('DenseNet121')

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
    log.debug(f"Model summary: {model.summary()}")

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

    test_ds = preprocessor.get_test_dataset()
 
    # 1. Input Schema
    # -----------------
    # Your input is a batch of images with shape (32, 240, 240, 3)
    # We use -1 to indicate that the batch size can vary.
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, IMAGE_SIZE, IMAGE_SIZE, 1), "image")])

    # 2. Output Schema - Multilabel binary classification head
    # ------------------
    # Your model outputs a list of two arrays. We need to define a schema for each.
    # Array 1: Shape (1, 3)
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, len(CLASSES_NAME)), "classification")])

    # 3. Model Signature
    # --------------------
    # Combine the input and output schemas into a ModelSignature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    mlflow.tensorflow.log_model(
        model,
        "my_model",
        signature=signature,
        code_paths=["src"],
    )
    y_true = np.array([y.astype(int) for _, y in test_ds.unbatch().as_numpy_iterator()])

    evaluate_model(model=model, 
                    test_ds=test_ds, 
                    y_true_labels=y_true, 
                    output_dir=OUPUT_DIR,
                    class_name=CLASSES_NAME) 


def evaluate_model(*,model:tf.keras.Model, 
                   test_ds:tf.data.Dataset, 
                   y_true_labels:np.ndarray, 
                   output_dir:Path,
                   class_name:list[str]) -> None:
    """Evaluates the model.

    Args:
        model: The model to evaluate.
        test_ds: The test dataset.
        y_true_labels: The true labels.
        y_true_bboxes: The true bounding boxes.
        cfg: The configuration object.
        class_name: The list of class names.
    """
    log = get_logger(__name__)
    log.info("Evaluating model...")

    results = model.evaluate(test_ds, return_dict=True)
    mlflow.log_dict(results, 'test_metrics.json')

    y_prob_pred = model.predict(test_ds)
    y_pred = (y_prob_pred>0.5).astype(int)

    report = classification_report(y_true_labels,
                                    y_pred, 
                                    target_names=class_name,
                                    output_dict=True)
    mlflow.log_dict(report, 'classification_report.json') 

    auc_fig = plot_auc_curve(output_dir=output_dir, 
                                    class_name_list=class_name, 
                                    y_true=y_true_labels, 
                                    y_prob_pred=y_pred)

    mlflow.log_figure(auc_fig, 'ROC-Curve.png')


    
    # log.info("Model evaluated.")
def get_callbacks(to_monitor, mode) -> list[tf.keras.callbacks.Callback]:
    
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
                                            mode=mode, patience=15, restore_best_weights=True),
    ]
    
    return callbacks

if __name__ == '__main__':
    main()





