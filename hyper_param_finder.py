# https://www.kaggle.com/code/mistag/keras-model-tuning-with-optuna#Objective-function
# https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_eager_simple.py
# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
# https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_eager_simple.py
# https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
from pathlib import Path

import mlflow
import optuna

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from hydra import compose, initialize
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

from utils.utils import setup_evnironment_vars
from utils.weighted_loss import get_weighted_loss

# https://www.tensorflow.org/guide/mixed_precision
mixed_precision.set_global_policy('mixed_float16')

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

LABELS =['Atelectasis','Effusion','Infiltration', 'Mass','Nodule']

# labels =['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
#     'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
#     'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


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
    label = tf.cast(label,tf.float32)
    return image, label

def get_preprocessed_dataset(batch_size):

    sample_df = pd.read_csv(CSV_PATH)
    valid_csv= 'datasets/valid-small.csv'
    valid_df = pd.read_csv(f"{valid_csv}")
    valid_df.drop('PatientId', axis=1, inplace=True)

    test_csv='datasets/test.csv'
    test_df = pd.read_csv(f"{test_csv}")
    test_df.drop('PatientId', axis=1, inplace=True)

    # Data Clean up
    # Droping duplicates
    sample_df.drop_duplicates(subset=['Patient ID'], inplace=True)
    sample_labels = ['Image Index', 'Finding Labels']
    sample_df = sample_df[sample_labels]

    # Prepare new csv file with useful information and format
    useful_data_df= sample_df[['Image Index', 'Finding Labels']]
    useful_data_df = useful_data_df.dropna()
    images_df = useful_data_df['Image Index'] 

    # One hot encoding
    train_cat_labels_df = useful_data_df['Finding Labels'].str.get_dummies(sep='|').astype('float32')
    train_cat_labels_df = train_cat_labels_df[LABELS]
    valid_images = valid_df.Image
    valid_labels_df = valid_df[LABELS]

    test_images = test_df.Image
    test_labels_df = test_df[LABELS]

    # Load Training Dataset from Dataframe
    train_ds = tf.data.Dataset.from_tensor_slices((images_df,  train_cat_labels_df.values))
    train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_images.values,  valid_labels_df.values))
    valid_ds = valid_ds.map(load_image_valid, num_parallel_calls=AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images.values,  test_labels_df.values))
    test_ds = test_ds.map(load_image_valid, num_parallel_calls=AUTOTUNE)

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
        # image = tf.keras.applications.densenet.preprocess_input(image)
        image = normalization_layer(image)
        return image, label


    train_ds = train_ds.map(lambda image, label: (data_augmentation(image), label) , num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(preprocess_image,num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).shuffle(len(images_df))
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    valid_ds = valid_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Class Imbalance Handling
    # Compute Class Frequencies
    N = train_cat_labels_df.shape[0]
    positive_frequencies = (train_cat_labels_df==1).sum()/N
    negative_frequencies = (train_cat_labels_df==0).sum()/N

    pos_weights = negative_frequencies.values
    neg_weights = positive_frequencies.values


    return train_ds, valid_ds, test_ds, pos_weights, neg_weights



# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")



def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    base_model = DenseNet121(
        include_top=False,
        weights='pretrain_weights/densenet.hdf5', #'imagenet', 
        input_shape= (IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    for i in range(n_layers):
        num_hidden = trial.suggest_int(name="n_units_l{}".format(i), 
                                       low= 64, high = 512, log=True)

        x = tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )(x)
        
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    # and a logistic layer
    x = Dense(len(LABELS), name='final_dense')(x)
    # activation must be float32 for metrics such as f1 score and so on
    predictions = Activation('sigmoid', dtype='float32', name='predictions')(x)

    return Model(inputs=base_model.input, outputs=predictions)

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer) # memory saving
    return optimizer


def objective(trial):
    # Clear clutter from previous session graphs.
    tf.keras.backend.clear_session()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_AUC', 
        patience=5, 
        mode='max', 
        restore_best_weights=True
    )

    # Hyperparameters to tune
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    with mlflow.start_run(nested=True):
        train_ds, valid_ds, test_ds, pos_weights, neg_weights = get_preprocessed_dataset(batch_size)
        mlflow.log_param('batch_size',batch_size)

        METRICS = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tfa.metrics.F1Score(num_classes=len(LABELS), average='macro'),
            tf.keras.metrics.AUC(name='AUC'), 
        ]
        model = create_model(trial)
        optimizer = create_optimizer(trial)
        model.compile(optimizer=optimizer, 
                    loss=get_weighted_loss(pos_weights, neg_weights),
                    metrics=METRICS)  
        
        model.fit(train_ds, 
                    validation_data=valid_ds,
                     callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_AUC'),
                                early_stopping],
                    epochs = NUM_EPOCHS)

        
    score = model.evaluate(test_ds, verbose=0)
    return score[-1]  # val_AUC is at last

def main()-> None:
    setup_evnironment_vars()

    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        raise Exception("No GPU found")
    else:
        print(f'[INFO] {found_gpu=}, {tf.__version__=}')


    mlflow.set_experiment("/hyper-param-tunning")
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[champion_callback])

    mlflow.log_params(study.best_params)
    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Chest X-ray with 4000 image for training",
            "optimizer_engine": "optuna",
            "model_family": "DenseNet121",
            "feature_set_version": 1,
        }
    )


if __name__ == '__main__':
    main()


