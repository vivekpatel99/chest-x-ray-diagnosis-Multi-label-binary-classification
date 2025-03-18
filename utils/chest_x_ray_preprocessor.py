import logging
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.logs import get_logger


class ChestXRayPreprocessor:
    """
    A class for preprocessing chest X-ray images and their labels.

    This class handles loading, cleaning, augmenting, normalizing, and preparing
    datasets for training and evaluation of chest X-ray classification models.
    It is optimized for memory efficiency by using generators and TensorFlow's
    data pipeline.
    """

    def __init__(self, config, labels: list | None = None) -> None:
        """
        Initializes the ChestXRayPreprocessor class.

        Args:
            config: Configuration object containing dataset directories and training parameters.
            labels (list, optional): List of labels for classification. Defaults to
                                     ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule'].
        """
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.config = config
        self.batch_size: int = config.TRAIN.BATCH_SIZE
        self.image_size: int = config.TRAIN.IMG_SIZE
        self.labels: list = labels or ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule']
        self.train_csv_labels: list = ['Image Index', 'Finding Labels']

        self.normalization_layer = tf.keras.layers.Normalization()
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.1, 0.1),
            tf.keras.layers.GaussianNoise(0.01)
        ])

    def _load_image(self, image_name: str, is_training: bool = True) -> tf.Tensor:
        """Loads and decodes a single image."""
        img_dir = self.config.DATASET_DIRS.TRAIN_IMAGES_DIR if is_training else self.config.DATASET_DIRS.TEST_IMAGE_DIR
        full_path = tf.strings.join([img_dir, '/', image_name])
        image = tf.io.read_file(full_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.keras.preprocessing.image.smart_resize(image, [self.image_size, self.image_size])
        return image

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Applies data augmentation to a single image."""
        return self.data_augmentation(image)

    def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """Normalizes a single image."""
        return self.normalization_layer(image)

    def _prepare_image_and_label(self, image_name: str, label: tf.Tensor, is_training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prepares a single image and its label for training or testing.

        Args:
            image_name (str): The name of the image file.
            label (tf.Tensor): The label for the image.
            is_training (bool): Whether the image is for training.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The preprocessed image and its label.
        """
        image = self._load_image(image_name, is_training)
        if is_training:
            image = self._augment_image(image)
        image = self._normalize_image(image)
        label = tf.cast(label, tf.float32)
        return image, label

    def _dataframe_generator(self, df: pd.DataFrame, is_training: bool) -> Tuple[str, tf.Tensor]:
        """
        Generates image names and labels from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing image names and labels.
            is_training (bool): Whether the data is for training.

        Yields:
            Tuple[str, tf.Tensor]: The image name and its corresponding label.
        """
        for _, row in df.iterrows():
            image_name = row['Image Index']
            label = tf.constant(row[self.labels].values, dtype=tf.float32)
            yield image_name, label

    def _create_dataset_from_generator(self, generator, is_training: bool) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset from a generator.

        Args:
            generator: The generator function.
            is_training (bool): Whether the data is for training.

        Returns:
            tf.data.Dataset: The created dataset.
        """
        output_signature = (tf.TensorSpec(shape=(), dtype=tf.string),
                            tf.TensorSpec(shape=(len(self.labels),), dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        dataset = dataset.map(lambda x, y: self._prepare_image_and_label(x, y, is_training),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            dataset = dataset.shuffle(buffer_size=dataset.cardinality())
        return dataset

    def _prepare_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        """
        Prepares a dataset for training or testing.

        Args:
            dataset (tf.data.Dataset): The dataset to prepare.
            batch_size (int): The batch size.

        Returns:
            tf.data.Dataset: The prepared dataset.
        """
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _train_df_clean_up(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans up the training dataframe.

        Args:
            train_df (pd.DataFrame): The raw training dataframe.

        Returns:
            pd.DataFrame: The cleaned training dataframe.
        """
        self.log.info("Cleaning up training dataframe")
        new_train_df = train_df[self.train_csv_labels]
        train_categorical_labels_df = new_train_df[self.train_csv_labels[1]].str.get_dummies(sep='|').astype('float32')
        train_images_df = new_train_df['Image Index']
        train_categorical_labels_df = train_categorical_labels_df[self.labels]
        return pd.concat([train_images_df, train_categorical_labels_df], axis=1)

    def _normalize_layer_adapt(self, train_ds: tf.data.Dataset) -> None:
        """
        Adapts the normalization layer to the training data.

        Args:
            train_ds (tf.data.Dataset): The training dataset.
        """
        images_for_stats = tf.concat([images for images, _ in train_ds.take(100)], axis=0)
        self.normalization_layer.adapt(images_for_stats)

    def load_and_preprocess_dataframe(self, csv_path: str, is_training: bool, split_ratio: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset:
        """
        Loads a dataframe from CSV, preprocesses it, and returns a tf.data.Dataset.

        Args:
            csv_path (str): Path to the CSV file.
            is_training (bool): Whether the dataset is for training.
            split_ratio (float): The ratio of the dataset to be used for validation.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset: If is_training is True, returns a tuple of (train_dataset, validation_dataset).
                                                                        Otherwise, returns the test_dataset.
        """
        self.log.info(f"Loading and preprocessing dataframe from {csv_path}")
        df = pd.read_csv(csv_path)
        df = self._train_df_clean_up(df)
        self.log.info(f"Loaded dataframe with shape: {df.shape} and {len(df)} rows")

        if is_training:
            train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=42, stratify=df[self.labels].values)

            train_generator = lambda: self._dataframe_generator(train_df, is_training=True)
            val_generator = lambda: self._dataframe_generator(val_df, is_training=False)

            train_dataset = self._create_dataset_from_generator(train_generator, is_training=True)
            val_dataset = self._create_dataset_from_generator(val_generator, is_training=False)

            return train_dataset, val_dataset
        else:
            test_generator = lambda: self._dataframe_generator(df, is_training=False)
            test_dataset = self._create_dataset_from_generator(test_generator, is_training=False)
            return test_dataset

    def get_training_and_validation_datasets(self, batch_size: int | None = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
        """
        Loads, preprocesses, and prepares training and validation datasets.

        Args:
            batch_size (int, optional): The batch size. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.Tensor, tf.Tensor]:
                A tuple containing the training dataset, the validation dataset,
                positive weights, and negative weights.
        """
        self.log.info(f"Getting training and validation datasets with batch size:{batch_size}")
        train_ds, valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TRAIN_CSV, is_training=True)

        self._normalize_layer_adapt(train_ds=train_ds)

        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        train_ds = self._prepare_dataset(train_ds, batch_size)
        valid_ds = self._prepare_dataset(valid_ds, batch_size)

        # Load training dataframe again to calculate class weights
        train_df = pd.read_csv(self.config.DATASET_DIRS.TRAIN_CSV)
        train_df = self._train_df_clean_up(train_df)
        pos_weights, neg_weights = self._get_class_weights(train_df)

        return train_ds, valid_ds, pos_weights, neg_weights

    def get_test_dataset(self, batch_size: int | None = None) -> tf.data.Dataset:
        """
        Loads, preprocesses, and prepares the test dataset.

        Args:
            batch_size (int, optional): The batch size. Defaults to None.

        Returns:
            tf.data.Dataset: The prepared test dataset.
        """
        self.log.info("Getting test dataset")
        test_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TEST_CSV, is_training=False)

        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        test_ds = self._prepare_dataset(test_ds, batch_size)
        return test_ds

    def _get_class_weights(self, labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates class weights.

        Args:
            labels_df (pd.DataFrame): The DataFrame containing the labels.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The positive and negative class weights.
        """
        N = labels_df.shape[0]
        positive_frequencies = (labels_df[self.labels] == 1).sum() / N
        negative_frequencies = (labels_df[self.labels] == 0).sum() / N
        pos_weights = negative_frequencies.values
        neg_weights = positive_frequencies.values
        return pos_weights, neg_weights

# class ChestXRayPreprocessor:
#     def __init__(self, config, labels:list|None=None) -> None:
#         """
#         Initializes the ChestXRayPreprocessor class with configuration settings.

#         Args:
#             config: Configuration object containing various settings for training and dataset directories.
#             labels (list, optional): List of labels to use for classification. If not provided, defaults to 
#                                     ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule'].

#         Attributes:
#             batch_size (int): The batch size for training, derived from the config.
#             dataset_len (int): The length of the dataset, initialized to 0.
#             LABELS (list): List of labels for classification.
#             TRAIN_CSV_LABELS (list): List of labels for the training CSV file.
#             normalization_layer (tf.keras.layers.Normalization): Normalization layer for preprocessing images.
#             data_augmentation (tf.keras.Sequential): Sequential model for data augmentation, including random 
#                                                     rotation, translation, and zoom.
#         """
#         self.log = get_logger(__name__, log_level=logging.INFO)
#         self.config = config
#         self.batch_size:int = config.TRAIN.BATCH_SIZE
#         self.dataset_len:int = 0 
#         self.image_size:int = config.TRAIN.IMG_SIZE
#         if labels:
#             self.LABELS = labels or ['Atelectasis','Effusion','Infiltration', 'Mass','Nodule']

#         self.TRAIN_CSV_LABELS = ['Image Index', 'Finding Labels']

#         self.normalization_layer = tf.keras.layers.Normalization()
#         self.data_augmentation = tf.keras.Sequential([
#             tf.keras.layers.RandomRotation(0.10),
#             tf.keras.layers.RandomTranslation(0.05, 0.05),
#             tf.keras.layers.RandomZoom(0.1, 0.1),
#             tf.keras.layers.GaussianNoise(0.01)  # Simulates quantum noise
#         ])


#     def load_image(self, image_name, label, is_training=True)-> tuple[tf.Tensor, tf.Tensor]:
#         """Loads and preprocesses an image."""
#         img_dir = self.config.DATASET_DIRS.TRAIN_IMAGES_DIR if is_training else self.config.DATASET_DIRS.TEST_IMAGE_DIR
#         full_path = tf.strings.join([img_dir, '/', image_name])
#         image = tf.io.read_file(full_path)
#         image = tf.io.decode_png(image, channels=1)
#         # image = tf.image.resize(image, 
#         #                         [self.image_size, self.image_size], 
#         #                         preserve_aspect_ratio=True,  
#         #                         antialias=True)
#         image = tf.keras.preprocessing.image.smart_resize(image, 
#                                 [self.image_size, self.image_size])
#         label = tf.cast(label, tf.float32)
#         return image, label

#     def augment_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
#         """Applies data augmentation to an image."""
#         self.log.info("Augmenting image")
#         return self.data_augmentation(image), label
    
#     def normalize_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
#         self.log.info("Normalizing image")
#         image = self.normalization_layer(image)
#         return image, label

#     def prepare_dataset(self, dataset, batch_size, is_training=True)-> tf.data.Dataset:
#         if is_training:
#             self.log.info("Preparing training dataset")
#             dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
         
#         dataset = dataset.map(self.normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.batch(batch_size)
#         return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#     def get_class_weights(self, labels_df)-> tuple[tf.Tensor, tf.Tensor]:
#         """Calculates class weights."""
#         N = labels_df.shape[0]
#         positive_frequencies = (labels_df == 1).sum() / N
#         negative_frequencies = (labels_df == 0).sum() / N
#         pos_weights = negative_frequencies.values
#         neg_weights = positive_frequencies.values
#         return pos_weights, neg_weights

#     def train_df_clean_up(self, train_df)-> tuple[pd.DataFrame, pd.DataFrame]:
#         """Cleans up the training dataframe."""
#         self.log.info("Cleaning up training dataframe")
#         new_train_df = train_df[self.TRAIN_CSV_LABELS]
#         train_categorical_labels_df = new_train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
#         train_images_df = new_train_df['Image Index'] 
#         train_categorical_labels_df = train_categorical_labels_df[self.LABELS]
#         return train_images_df, train_categorical_labels_df
#         # new_train_df = train_df[self.LABELS]
#         # # train_categorical_labels_df = new_train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
#         # train_images_df = train_df['Image'] 
#         # train_categorical_labels_df = new_train_df[self.LABELS]
#         # return train_images_df, train_categorical_labels_df


#     def _normlization_layer_adapt(self, train_ds:tf.data.Dataset) -> None:
#         """Adapts the normalization layer to the training data."""
#         # images_for_stats =  tf.concat([images for images, _ in train_ds.take(int(dataset_len *0.30))], axis=0) 
#         images_for_stats =  tf.concat([images for images, _ in train_ds.as_numpy_iterator()], axis=0) 
#         self.normalization_layer.adapt(images_for_stats)
#         # self.normalization_layer.adapt(train_ds.map(lambda x, y: x))    
#     def old_load_and_preprocess_dataframe(self, csv_path: str, is_training: bool) -> tf.data.Dataset:
#         """Loads a dataframe from CSV, preprocesses it, and returns a tf.data.Dataset."""
#         self.log.info("Loading and preprocessing dataframe")
#         limit=4000
#         df = pd.read_csv(csv_path)#[:limit]
#         if is_training:
#             images_df, labels_df = self.train_df_clean_up(df)
#         else:
#             images_df = df['Image']
#             labels_df = df[self.LABELS]
#         self.log.info(f"Loaded dataframe with shape: {df.shape} and {len(df)} rows")
#         dataset = tf.data.Dataset.from_tensor_slices((images_df.values, labels_df.values))
#         dataset = dataset.map(lambda x, y: self.load_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
#         if is_training:
#             dataset = dataset.shuffle(buffer_size=len(df))  # Only shuffle training data
#         return dataset
    
#     def load_and_preprocess_dataframe(self, csv_path: str, is_training: bool, split_ratio: float = 0.2) -> tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset:
#         """Loads a dataframe from CSV, preprocesses it, and returns a tf.data.Dataset.
#         Args:
#             csv_path (str): Path to the CSV file.
#             is_training (bool): Whether the dataset is for training.
#             split_ratio (float): The ratio of the dataset to be used for validation.
#         Returns:
#             tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset: If is_training is True, returns a tuple of (train_dataset, validation_dataset).
#                                                                         Otherwise, returns the test_dataset.
#         """
#         self.log.info("Loading and preprocessing dataframe")
#         df = pd.read_csv(csv_path)
#         images_df, labels_df = self.train_df_clean_up(df)

#         self.log.info(f"Loaded dataframe with shape: {df.shape} and {len(df)} rows")

#         if is_training:
#             # Split the data into training and validation sets
#             train_images, val_images, train_labels, val_labels = train_test_split(
#                 images_df.values, labels_df.values, test_size=split_ratio, random_state=42
#             )

#             train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#             val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

#             train_dataset = train_dataset.map(lambda x, y: self.load_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
#             val_dataset = val_dataset.map(lambda x, y: self.load_image(x, y, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)

#             train_dataset = train_dataset.shuffle(buffer_size=len(train_images))
#             return train_dataset, val_dataset
#         else:
#             dataset = tf.data.Dataset.from_tensor_slices((images_df.values, labels_df.values))
#             dataset = dataset.map(lambda x, y: self.load_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
#             return dataset
    
#     def get_training_and_validation_datasets(self, batch_size: int | None = None) -> tuple[
#         tf.data.Dataset, tf.data.Dataset, tf.Tensor, tf.Tensor]:
#         """Loads, preprocesses, and prepares training and validation datasets.

#         Returns:
#             A tuple containing the training dataset, the validation dataset, positive weights, and negative weights.
#         """
#         self.log.info(f"Getting training and validation datasets with batch size:{batch_size}")
#         train_ds, valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TRAIN_CSV, is_training=True)
#         # valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.VALID_CSV, is_training=False)

#         self._normlization_layer_adapt(train_ds=train_ds)

#         if not batch_size:
#             batch_size = self.config.TRAIN.BATCH_SIZE

#         train_ds = self.prepare_dataset(train_ds, batch_size, is_training=True)
#         valid_ds = self.prepare_dataset(valid_ds, batch_size, is_training=False)

#         # Load training dataframe again to calculate class weights
#         train_df = pd.read_csv(self.config.DATASET_DIRS.TRAIN_CSV)
#         _, train_categorical_labels_df = self.train_df_clean_up(train_df)
#         pos_weights, neg_weights = self.get_class_weights(train_categorical_labels_df)

#         return train_ds, valid_ds, pos_weights, neg_weights


#     def get_test_dataset(self, batch_size: int | None = None) -> tf.data.Dataset:
#         """Loads, preprocesses, and prepares the test dataset."""
#         self.log.info("Getting test dataset")
#         test_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TEST_CSV, is_training=False)

#         if not batch_size:
#             batch_size = self.config.TRAIN.BATCH_SIZE

#         test_ds = self.prepare_dataset(test_ds, batch_size, is_training=False)
#         return test_ds
