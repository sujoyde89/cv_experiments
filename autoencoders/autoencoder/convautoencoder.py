from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, \
									Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
	@staticmethod
	def build(height, width, depth, filters=(32, 64, 96), latent_dim=64):
		input_shape = (height, width, depth)
		chanDim = -1

		# encoder
		inputs = Input(shape = input_shape)
		x = inputs
		for f in filters:
			x = Conv2D(f, (3,3), strides = 2, padding = 'same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		volume_size = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latent_dim)(x)
		encoder = Model(inputs, latent, name="encoder")

		# decoder
		latent_inputs = Input(shape=(latent_dim,))
		x = Dense(np.prod(volume_size[1:]))(latent_inputs)
		x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)
		decoder = Model(latent_inputs, outputs, name="decoder")

		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")

		return (encoder, decoder, autoencoder)