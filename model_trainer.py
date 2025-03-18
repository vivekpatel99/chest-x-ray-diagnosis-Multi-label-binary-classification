
import logging
import os
from enum import Enum

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra import compose, initialize
from matplotlib import pyplot as plt
from sklearn import metrics

from model import build_DenseNet121
from utils.chest_x_ray_preprocessor import ChestXRayPreprocessor
from utils.logs import get_logger
from utils.utils import setup_evnironment_vars
from utils.weighted_loss import get_weighted_loss

# https://www.tensorflow.org/guide/mixed_precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)


def main()-> None:
    log = get_logger(__name__, log_level=logging.INFO)
    setup_evnironment_vars()
    log.info('setup_evnironment_vars done')

    # https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")

    BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    IMAGE_SIZE = cfg.TRAIN.IMG_SIZE
    LEARNING_RATE = cfg.TRAIN.LEARNING_RATE
    NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
    LOG_DIR = cfg.OUTPUTS.LOG_DIR
    CHECK_POINT_DIR = cfg.OUTPUTS.CHECKPOINT_PATH
    # INPUT_SHAPE = cfg.TRAIN.INPUT_SHAPE
    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.error("No GPU found")
        raise Exception("No GPU found")
    else:
        log.info(f'{found_gpu=}, {tf.__version__=}')
 
    labels =['Atelectasis','Effusion','Infiltration','Nodule']
    preprocessor = ChestXRayPreprocessor(cfg, labels=labels)
    train_ds, valid_ds, pos_weights, neg_weights = preprocessor.get_training_and_validation_datasets(BATCH_SIZE)

    # Model Development
    model = build_DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), num_classes=len(labels))

    METRICS = [
        'accuracy',
        'binary_accuracy',
        tf.keras.metrics.AUC(name='AUC',
                             multi_label=True), 
        tf.keras.metrics.F1Score(average='weighted'),
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                loss=get_weighted_loss(pos_weights, neg_weights),
                metrics=METRICS)     


    to_monitor = 'val_loss'
    mode = 'min'
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, 
                                             patience=2, 
                                             monitor=to_monitor,
                                             mode=mode,
                                             min_lr=1e-6,
                                             verbose=1),

        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(str(CHECK_POINT_DIR), "ckpt_{epoch}") ,
                                            save_weights_only=False,
                                            save_best_only=True,
                                            monitor=to_monitor,
                                            mode=mode,
                                            verbose=1),
        # tf.keras.callbacks.EarlyStopping(monitor=MONITOR[0], 
        #                                 patience=10,
        #                                 mode=MONITOR[0], 
        #                                 restore_best_weights=True),

    ]
    
    mlflow.set_experiment("/chest_xray_training_densenet121")

    # mlflow.log_params({
    #     "batch_size": BATCH_SIZE,
    #     "image_size": IMAGE_SIZE,
    #     "learning_rate": LEARNING_RATE,
    # })
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)
    model.fit(train_ds, 
            validation_data=valid_ds,
            epochs = NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks)
    
    test_ds = preprocessor.get_test_dataset(BATCH_SIZE)
    y_scores = model.predict(test_ds)

    test_df = pd.read_csv(cfg.DATASET_DIRS.TEST_CSV)
    y_true = test_df[labels].values
    y_scores = np.where(y_scores>0.5, 1, 0)
    auc_roc_values = []
    fig, axs = plt.subplots(1)
    for i in range(len(labels)):
        try:
            roc_score_per_label = metrics.roc_auc_score(y_true=y_true[:,i], y_score=y_scores[:,i])
            auc_roc_values.append(roc_score_per_label)
            fpr, tpr, _ = metrics.roc_curve(y_true=y_true[:,i],  y_score=y_scores[:,i])
            
            axs.plot([0,1], [0,1], 'k--')
            axs.plot(fpr, tpr, 
                    label=f'{labels[i]} - AUC = {round(roc_score_per_label, 3)}')

            axs.set_xlabel('False Positive Rate')
            axs.set_ylabel('True Positive Rate')
            axs.legend(loc='lower right')
        except:
            log.error(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.savefig(f"{cfg.OUTPUTS.OUPUT_DIR}/ROC-Curve.png")
    mlflow.log_figure(fig, 'ROC-Curve.png')
    results = model.evaluate(test_ds, verbose=0,return_dict=True)
    mlflow.log_metrics(results)

if __name__ == '__main__':
    main()


