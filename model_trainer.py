
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
import os
from math import exp
from pathlib import Path

import mlflow
import mlflow.experiments
import numpy as np
from hydra import compose, initialize
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.metrics import classification_report
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers.schedules import CosineDecay

from src.data_loader.chest_x_ray_preprocessor import ChestXRayPreprocessor
from src.model.densenet121_model import build_DenseNet121
from src.model.ResNet50V2_model import build_ResNet50V2
from src.utils.logs import get_logger
from src.utils.utils import download_dataset, plot_auc_curve
from src.weighted_loss.focal_loss import focal_loss
from src.weighted_loss.weighted_loss import get_weighted_loss

# mixed_precision.set_global_policy('mixed_float16')

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
    
# Constants
IMAGE_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE
OUPUT_DIR = Path(cfg.OUTPUTS.OUPUT_DIR)
OUPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECK_POINT_DIR = Path(cfg.OUTPUTS.CHECKPOINT_PATH)
CHECK_POINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = 'logs'
EXPERIMENT_NAME = 'build_DenseNet121-focal_loss'



def main() -> None:
    log = get_logger(__name__, log_level=logging.INFO)

    mlflow.set_experiment(EXPERIMENT_NAME) 

    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.info("No GPU found")
        raise Exception("No GPU found")
    
    # Look into the data directory
    datasets = 'datasets/sample'
    dataset_path = Path(datasets)
    download_dataset(log, dataset_path)

    CLASSES_NAME = ['Atelectasis','Effusion','Infiltration', 'Mass', 'No Finding']

    preprocessor = ChestXRayPreprocessor(cfg, labels=CLASSES_NAME)
    train_ds, valid_ds, pos_weights, neg_weights, steps_per_epoch= preprocessor.get_training_and_validation_datasets()

    to_monitor = 'val_AUC'
    mode = 'max'
    mlflow.tensorflow.autolog(log_models=True, 
                            log_datasets=False, 
                            log_input_examples=True,
                            log_model_signatures=True,
                            keras_model_kwargs={"save_format": "keras"},
                            checkpoint_monitor=to_monitor, 
                            checkpoint_mode=mode)
    # Start an MLflow run
    with mlflow.start_run() as _:
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
        model = build_DenseNet121(input_shape=input_shape, num_classes=len(CLASSES_NAME))
        # model = build_ResNet50V2(input_shape=input_shape, num_classes=len(CLASSES_NAME))
        log.debug(f"Model summary: {model.summary()}")

        METRICS = [
            'binary_accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC'), 
        ]
        lr_schedule = CosineDecay(
            LEARNING_RATE,
            decay_steps=steps_per_epoch * NUM_EPOCHS,
            alpha=0.001
        )
        model.compile(optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=lr_schedule,
                        weight_decay=1e-5,
                        global_clipnorm=1.0), 
                    # loss=get_weighted_loss(pos_weights, neg_weights),
                    loss=focal_loss,
                    metrics=METRICS)     

        callbacks = get_callbacks(to_monitor, mode)

        model.fit(train_ds, 
                steps_per_epoch=steps_per_epoch,
                validation_data=valid_ds,
                batch_size=BATCH_SIZE,
                epochs = NUM_EPOCHS,
                callbacks=callbacks)

        y_true = np.array([y.astype(int) for _, y in valid_ds.unbatch().as_numpy_iterator()])

        evaluate_model(log=log,
                        model=model, 
                        test_ds=valid_ds, 
                        y_true_labels=y_true, 
                        output_dir=OUPUT_DIR,
                        class_name=CLASSES_NAME)



def evaluate_model(*,log,
                   model:tf.keras.Model, 
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

    log.info("Model evaluated.")

def get_callbacks(to_monitor, mode) -> list[tf.keras.callbacks.Callback]:
    
    checkpoint_prefix = str(CHECK_POINT_DIR / f"best_{EXPERIMENT_NAME}.keras")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                            save_best_only=True, # Save only the best model based on val_loss
                                            monitor=to_monitor,
                                            mode=mode,
                                            verbose=1),  # Display checkpoint saving messages
        tf.keras.callbacks.ReduceLROnPlateau(monitor=to_monitor,
                                            mode=mode,
                                            verbose=1,
                                            factor=0.1, patience=5, min_lr=1e-8),
        tf.keras.callbacks.EarlyStopping(monitor=to_monitor,
                                            mode=mode,
                                            verbose=1, 
                                            patience=10, restore_best_weights=True),
    ]
    
    return callbacks

if __name__ == '__main__':
    main()





