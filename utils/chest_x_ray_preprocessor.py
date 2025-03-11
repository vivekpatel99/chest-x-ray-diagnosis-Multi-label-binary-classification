import logging

import pandas as pd
import tensorflow as tf

from utils.logs import get_logger


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
        if labels:
            self.LABELS = labels or ['Atelectasis','Effusion','Infiltration', 'Mass','Nodule']

        self.TRAIN_CSV_LABELS = ['Image Index', 'Finding Labels']

        self.normalization_layer = tf.keras.layers.Normalization()
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.1, 0.1),
            tf.keras.layers.GaussianNoise(0.01)  # Simulates quantum noise
        ])


    def load_image(self, image_name, label, is_training=True)-> tuple[tf.Tensor, tf.Tensor]:
        """Loads and preprocesses an image."""
        img_dir = self.config.DATASET_DIRS.TRAIN_IMAGES_DIR if is_training else self.config.DATASET_DIRS.TEST_IMAGE_DIR
        full_path = tf.strings.join([img_dir, '/', image_name])
        image = tf.io.read_file(full_path)
        image = tf.io.decode_png(image, channels=1)
        # image = tf.image.resize(image, 
        #                         [self.image_size, self.image_size], 
        #                         preserve_aspect_ratio=True,  
        #                         antialias=True)
        image = tf.keras.preprocessing.image.smart_resize(image, 
                                [self.image_size, self.image_size])
        label = tf.cast(label, tf.float32)
        return image, label

    def augment_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
        """Applies data augmentation to an image."""
        self.log.info("Augmenting image")
        return self.data_augmentation(image), label
    
    def normalize_image(self, image, label)-> tuple[tf.Tensor, tf.Tensor]:
        self.log.info("Normalizing image")
        image = self.normalization_layer(image)
        return image, label

    def prepare_dataset(self, dataset, batch_size, is_training=True)-> tf.data.Dataset:
        if is_training:
            self.log.info("Preparing training dataset")
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
         
        dataset = dataset.map(self.normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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
        new_train_df = train_df[self.TRAIN_CSV_LABELS]
        train_categorical_labels_df = new_train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
        train_images_df = new_train_df['Image Index'] 
        train_categorical_labels_df = train_categorical_labels_df[self.LABELS]
        return train_images_df, train_categorical_labels_df
        # new_train_df = train_df[self.LABELS]
        # # train_categorical_labels_df = new_train_df[self.TRAIN_CSV_LABELS[1]].str.get_dummies(sep='|').astype('float32')
        # train_images_df = train_df['Image'] 
        # train_categorical_labels_df = new_train_df[self.LABELS]
        # return train_images_df, train_categorical_labels_df


    def _normlization_layer_adapt(self, train_ds:tf.data.Dataset) -> None:
        """Adapts the normalization layer to the training data."""
        # images_for_stats =  tf.concat([images for images, _ in train_ds.take(int(dataset_len *0.30))], axis=0) 
        images_for_stats =  tf.concat([images for images, _ in train_ds.as_numpy_iterator()], axis=0) 
        self.normalization_layer.adapt(images_for_stats)
        # self.normalization_layer.adapt(train_ds.map(lambda x, y: x))    
    def load_and_preprocess_dataframe(self, csv_path: str, is_training: bool) -> tf.data.Dataset:
        """Loads a dataframe from CSV, preprocesses it, and returns a tf.data.Dataset."""
        self.log.info("Loading and preprocessing dataframe")
        limit=4000
        df = pd.read_csv(csv_path)#[:limit]
        if is_training:
            images_df, labels_df = self.train_df_clean_up(df)
        else:
            images_df = df['Image']
            labels_df = df[self.LABELS]
        self.log.info(f"Loaded dataframe with shape: {df.shape} and {len(df)} rows")
        dataset = tf.data.Dataset.from_tensor_slices((images_df.values, labels_df.values))
        dataset = dataset.map(lambda x, y: self.load_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(df))  # Only shuffle training data
        return dataset
    
    def get_training_and_validation_datasets(self, batch_size: int | None = None) -> tuple[
        tf.data.Dataset, tf.data.Dataset, tf.Tensor, tf.Tensor]:
        """Loads, preprocesses, and prepares training and validation datasets.

        Returns:
            A tuple containing the training dataset, the validation dataset, positive weights, and negative weights.
        """
        self.log.info(f"Getting training and validation datasets with batch size:{batch_size}")
        train_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TRAIN_CSV, is_training=True)
        valid_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.VALID_CSV, is_training=False)

        self._normlization_layer_adapt(train_ds=train_ds)

        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        train_ds = self.prepare_dataset(train_ds, batch_size, is_training=True)
        valid_ds = self.prepare_dataset(valid_ds, batch_size, is_training=False)

        # Load training dataframe again to calculate class weights
        train_df = pd.read_csv(self.config.DATASET_DIRS.TRAIN_CSV)
        _, train_categorical_labels_df = self.train_df_clean_up(train_df)
        pos_weights, neg_weights = self.get_class_weights(train_categorical_labels_df)

        return train_ds, valid_ds, pos_weights, neg_weights


    def get_test_dataset(self, batch_size: int | None = None) -> tf.data.Dataset:
        """Loads, preprocesses, and prepares the test dataset."""
        self.log.info("Getting test dataset")
        test_ds = self.load_and_preprocess_dataframe(self.config.DATASET_DIRS.TEST_CSV, is_training=False)

        if not batch_size:
            batch_size = self.config.TRAIN.BATCH_SIZE

        test_ds = self.prepare_dataset(test_ds, batch_size, is_training=False)
        return test_ds
