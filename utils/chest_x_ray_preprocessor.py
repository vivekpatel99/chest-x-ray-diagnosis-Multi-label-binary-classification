import pandas as pd
import tensorflow as tf

class ChestXRayPreprocessor:
    def __init__(self, config):
        self.config = config
        self.batch_size:int = config.BATCH_SIZE
        self.dataset_len:int = 0 
        self.normalization_layer = tf.keras.layers.Normalization()
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomZoom(0.1, 0.1)
        ])
    def _normlization_layer_adapt(self, dataset_len:int, train_ds:tf.data):
        images_for_stats =  tf.concat([images for images, _ in train_ds.take(int(dataset_len *0.25))], axis=0) 
        self.normalization_layer.adapt(images_for_stats)

    def load_image(self, image_name, label, is_training=True):
        img_dir = self.config.TRAIN_IMG_DIR if is_training else 'datasets/images-small/'
        full_path = tf.strings.join([img_dir, '/', image_name])
        image = tf.io.read_file(full_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.config.IMAGE_SIZE, self.config.IMAGE_SIZE])
        label = tf.cast(label, tf.float32)
        return image, label

    def preprocess_image(self, image, label):
        image = self.normalization_layer(image)
        return image, label

    def augment_image(self, image, label):
        return self.data_augmentation(image), label

    def prepare_dataset(self, dataset, batch_size, is_training=True):
        if is_training:
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
         
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_class_weights(self, labels_df):
        N = labels_df.shape[0]
        positive_frequencies = (labels_df == 1).sum() / N
        negative_frequencies = (labels_df == 0).sum() / N
        pos_weights = negative_frequencies.values
        neg_weights = positive_frequencies.values
        return pos_weights, neg_weights

    def get_preprocessed_datasets(self, batch_size=int|None):
        train_df, valid_df, test_df = self._load_dataframes()
        train_ds = tf.data.Dataset.from_tensor_slices((train_df['Image Index'], train_df[self.config.LABELS].values))
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_df['Image'], valid_df[self.config.LABELS].values))
        test_ds = tf.data.Dataset.from_tensor_slices((test_df['Image'], test_df[self.config.LABELS].values))

        train_ds = train_ds.map(lambda x, y: self.load_image(x, y, True), num_parallel_calls=tf.data.AUTOTUNE)
        self._normlization_layer_adapt(dataset_len=len(train_df),train_ds=train_ds)
        valid_ds = valid_ds.map(lambda x, y: self.load_image(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: self.load_image(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
        
        if not batch_size:
            batch_size = self.config.BATCH_SIZE

        train_ds = self.prepare_dataset(train_ds, batch_size, is_training=True)
        valid_ds = self.prepare_dataset(valid_ds, batch_size, is_training=False)
        test_ds = self.prepare_dataset(test_ds, batch_size, is_training=False)

        pos_weights, neg_weights = self.get_class_weights(train_df[self.config.LABELS])

        return train_ds, valid_ds, test_ds, pos_weights, neg_weights

    def _load_dataframes(self):
        # Load and preprocess your dataframes here
        # This is a placeholder implementation
        train_df = pd.read_csv(self.config.CSV_PATH)
        valid_df = pd.read_csv(self.config.VALID_CSV)
        test_df = pd.read_csv(self.config.TEST_CSV)
        
        return train_df, valid_df, test_df
