from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
import efficientnet.keras as efn

from tensorflow.keras import backend as K
import numpy as np

class EfficientNetB:
    @staticmethod
    def build(height, width, depth, num_classes):
        input_shape = (height, width, depth)
        chanDim = -1
        
        model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        x=Flatten()(model.output)
        x = Dense(500)(x)
        x = Activation("relu")(x)
        #x = Dense(100)(x)
        #x = Activation("relu")(x)
        x = Dense(num_classes)(x)
        output = Activation("softmax")(x)
        model=Model(model.input,output, name='efficientnetb0')

        return model