
import math

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2, l1, l2


def build_DenseNet121(input_shape:tuple, num_classes:int):
    inputs = tf.keras.Input(shape=input_shape)
    inputs = tf.keras.applications.densenet.preprocess_input(inputs)
    base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
        )
    base_model.trainable = True

    total_layers = len(base_model.layers)
    trainable_layers = math.ceil(total_layers * 0.20) 
    # Then freeze all layers except the last layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=L1L2(0.001))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=L1L2(0.001))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(units=int(num_classes), name='final_dense',
              kernel_regularizer=L1L2(0.01))(x)
    
    # activation must be float32 for metrics such as f1 score and so on
    predictions = Activation('sigmoid', dtype='float32', name='predictions')(x)

    return  Model(inputs=base_model.input, outputs=predictions)
