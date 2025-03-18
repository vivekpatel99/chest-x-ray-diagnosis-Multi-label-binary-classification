import logging

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.logs import get_logger


class ChestXRayPreprocessor:
    def __init__(self, config, labels:list|None=None) -> None:
        """
        Initializes the ChestXRayPreprocessor class with configuration settings.

        Args:
            config: Configuration object containing various settings for training and dataset directories.
            labels (list, optional): List of labels to use for classification. If not provided, defaults to 
                                    ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule'].

        Attributes:
            batch_size (int): The batch size for training, derived from the config.
            dataset_len (int): The length of the dataset, initialized to 0.
            LABELS (list): List of labels for classification.
            TRAIN_CSV_LABELS (list): List of labels for the training CSV file.
            normalization_layer (tf.keras.layers.Normalization): Normalization layer for preprocessing images.
            data_augmentation (tf.keras.Sequential): Sequential model for data augmentation, including random 
                                                    rotation, translation, and zoom.
        """
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.config = config
        self.batch_size:int = config.TRAIN.BATCH_SIZE
        self.dataset_len:int = 0 
        self.image_size:int = config.TRAIN.IMG_SIZE
        self.pos_weights = None
        self.neg_weights = None
        self.LABELS = labels or ['Atelectasis','Effusion','Infiltration', 'Mass','Nodule']

        self.TRAIN_CSV_LABELS = ['Image Index', 'Finding Labels']

        self.normalization_layer = tf.keras.layers.Normalization()
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.1, 0.1),
            tf.keras.layers.GaussianNoise(0.01)  # Simulates quantum noise
        ])

    @tf.function
    def load_image(self, image_name, label, is_training=True)-> tuple[tf.Tensor, tf.Tensor]:
        """Loads and preprocesses an image."""
        img_dir = self.config.DATASET_DIRS.TRAIN_IMAGES_DIR if is_training else self.config.DATASET_DIRS.TEST_IMAGE_DIR
        full_path = tf.strings.join([img_dir, '/', image_name])
        image = tf.io.read_file(full_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.keras.preprocessing.image.smart_resize(image, 
                                [self.image_size, self.image_size])

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        label = tf.cast(label, tf.float32)
        return image, label
    
    @tf.function
    def augment_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
        """Applies data augmentation to an image."""
        self.log.info("Augmenting image")
        return self.data_augmentation(image), label
    
    @tf.function
    def normalize_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
        self.log.info("Normalizing image")
        image = self.normalization_layer(image)
        return image, label

    def prepare_dataset(self, dataset, batch_size, is_training=True)-> tf.data.Dataset:
        if is_training:
            self.log.info("Preparing training dataset with augmentation")
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(self.normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_class_weights(self, labels_df)-> tuple[tf.Tensor, tf.Tensor]:
        """Calculates class weights."""
        N = labels_df.shape[0]
        positive_frequencies = (labels_df == 1).sum() / N
        negative_frequencies = (labels_df == 0).sum() / N
        pos_weights = negative_frequencies.values
        neg_weights = positive_frequencies.values
        return pos_weights, neg_weights

    def train_df_clean_up(self, train_df)-> tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans up the training dataframe."""
        self.log.info("Cleaning up training dataframe")
        train_categorical_labels_df = train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
        train_images_df = train_df['Image Index'] 
        train_categorical_labels_df = train_categorical_labels_df[self.LABELS]
        self.pos_weights, self.neg_weights = self.get_class_weights(train_categorical_labels_df)
        return train_images_df, train_categorical_labels_df
        # new_train_df = train_df[self.LABELS]
        # # train_categorical_labels_df = new_train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
        # train_images_df = train_df['Image'] 
        # train_categorical_labels_df = new_train_df[self.LABELS]
        # return train_images_df, train_categorical_labels_df


    def _normlization_layer_adapt(self, train_ds:tf.data.Dataset) -> None:
        """Adapts the normalization layer to the training data."""
        # images_for_stats =  tf.concat([images for images, _ in train_ds.take(int(dataset_len *0.30))], axis=0) 
        # images_for_stats =  tf.concat([images for images, _ in train_ds.as_numpy_iterator()], axis=0) 
        images_for_stats = train_ds.map(lambda x, y: x)\
                                   .unbatch()\
                                   .batch(self.batch_size * 10)\
                                   .take(self.batch_size * 10)
        self.normalization_layer.adapt(images_for_stats)
        # self.normalization_layer.adapt(train_ds.map(lambda x, y: x))    

    
    def load_and_preprocess_dataframe(self, csv_path: str, is_training: bool, split_ratio: float = 0.2) -> tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset:
        """Loads a dataframe from CSV, preprocesses it, and returns a tf.data.Dataset.
        Args:
            csv_path (str): Path to the CSV file.
            is_training (bool): Whether the dataset is for training.
            split_ratio (float): The ratio of the dataset to be used for validation.
        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset: If is_training is True, returns a tuple of (train_dataset, validation_dataset).
                                                                        Otherwise, returns the test_dataset.
        """
        self.log.info("Loading and preprocessing dataframe")

        df = pd.read_csv(csv_path, usecols=self.TRAIN_CSV_LABELS)

        images_df, labels_df = self.train_df_clean_up(df)

        self.log.info(f"Loaded dataframe with shape: {df.shape} and {len(df)} rows")

        if is_training:
            # Split the data into training and validation sets
            train_images, val_images, train_labels, val_labels = train_test_split(images_df.values, 
                                                                                  labels_df.values, 
                                                                                  test_size=split_ratio, 
                                                                                  random_state=42, 
                                                                                  stratify=labels_df.values)

            self.log.info(f"training split: {len(train_images)} and validation split: {len(val_images)}")

            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

            train_dataset = train_dataset.map(lambda x, y: self.load_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = val_dataset.map(lambda x, y: self.load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

            train_dataset = train_dataset.shuffle(buffer_size=len(train_images))
            return train_dataset, val_dataset
        else:
            dataset = tf.data.Dataset.from_tensor_slices((images_df.values, labels_df.values))
            dataset = dataset.map(lambda x, y: self.load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
            return dataset
    
    def get_training_and_validation_datasets(self, batch_size: int | None = None) -> tuple[
        tf.data.Dataset, tf.data.Dataset, tf.Tensor| None, tf.Tensor | None]:
        """Loads, preprocesses, and prepares training and validation datasets.

        Returns:
            A tuple containing the training dataset, the validation dataset, positive weights, and negative weights.
        """
        self.log.info(f"Getting training and validation datasets with batch size:{batch_size}")
        train_ds, valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TRAIN_CSV, is_training=True)
        # valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.VALID_CSV, is_training=False)

        self._normlization_layer_adapt(train_ds=train_ds)

        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        train_ds = self.prepare_dataset(train_ds, batch_size, is_training=True)
        valid_ds = self.prepare_dataset(valid_ds, batch_size, is_training=False)

        return train_ds, valid_ds, self.pos_weights, self.neg_weights 


    def get_test_dataset(self, batch_size: int | None = None) -> tf.data.Dataset:
        """Loads, preprocesses, and prepares the test dataset."""
        self.log.info("Getting test dataset")
        df = pd.read_csv(self.config.DATASET_DIRS.TEST_CSV)

        image = df.Image
        labels_df = df[self.LABELS]

        dataset = tf.data.Dataset.from_tensor_slices((image.values, labels_df.values))
        test_ds = dataset.map(lambda x, y: self.load_image(x, y, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)
        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        test_ds = self.prepare_dataset(test_ds, batch_size, is_training=False)
        return test_ds