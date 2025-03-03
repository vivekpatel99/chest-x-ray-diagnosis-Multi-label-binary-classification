
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_DenseNet121(image_size:int, num_classes:int):
    base_model = DenseNet121(
            include_top=False,
            weights='pretrain_weights/densenet.hdf5', #'imagenet', 
            input_shape=(image_size, image_size, 3)  
        )
    # base_model.trainable = False


    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # and a logistic layer
    predictions = Dense(num_classes, activation="sigmoid")(x)

    return  Model(inputs=base_model.input, outputs=predictions)
