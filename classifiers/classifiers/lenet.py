from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
import numpy as np

class LeNet:
	@staticmethod
	def build(height, width, depth, num_classes, filters=(20, 50)):
		input_shape = (height, width, depth)
		chanDim = -1

		# encoder
		inputs = Input(shape = input_shape)
		x = inputs
		for f in filters:
			x = Conv2D(f, (5,5), padding = 'same')(x)
			x = Activation("relu")(x)
			x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

		#volume_size = K.int_shape(x)
		x = Flatten()(x)
		x = Dense(500)(x)
		x = Activation("relu")(x)
		x = Dense(num_classes)(x)
		output = Activation("softmax")(x)
		lenet = Model(inputs, output, name="lenet")

		return lenet