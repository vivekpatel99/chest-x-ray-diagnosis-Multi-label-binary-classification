
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_DenseNet121(input_shape:tuple, num_classes:int):
    base_model = DenseNet121(
            include_top=False,
            weights='imagenet', # input will be grayscale images
            input_shape=input_shape  
        )
    # base_model.trainable = False
    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    x = Dense(units=int(num_classes), name='final_dense')(x)
    # activation must be float32 for metrics such as f1 score and so on
    predictions = Activation('sigmoid', dtype='float32', name='predictions')(x)

    return  Model(inputs=base_model.input, outputs=predictions)
