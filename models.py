from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    BatchNormalization
)
from tensorflow.keras import regularizers



def resnet50_softmax(cfgData, num_classes):

    """

    Creates a CNN using the the ResNet50 architecture as backbone.
    The top layers are removed and replaced by a Dense layer with N neurons,
    which represent the number of classes to be identified by the network.

    The activation function of the last Dense layer is softmax.

    """

    IMAGE_SIZE = [cfgData['inputSize'], cfgData['inputSize']]
    # ResNet50 backbone
    backbone = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    x = BatchNormalization()(backbone.output)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=cfgData['dropout-rate'])(x)

    x = Flatten()(x)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(num_classes, activation='softmax',
                       activity_regularizer=regularizers.l2(cfgData['scale-l2-regularizer']))(x)

    # create a model object
    model = Model(inputs=backbone.input, outputs=prediction)

    return model, preprocess_input_resnet50
