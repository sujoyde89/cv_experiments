import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow as tf

from get_data import DataFetcher
from autoencoder.convautoencoder import ConvAutoencoder
import config

seed(12)
tf.random.set_seed(12)

def  main():
	epochs = config.EPOCHS
	bs = config.BS
	plot_file = config.HISTORY_FILE
	target_size = config.TARGET_SIZE
	samples = config.SAMPLES
	is_sub_dir = config.IS_SUB_DIR
	load_by_cv = config.LOAD_BY_CV

	height = target_size[0]
	width = target_size[1]
	depth = target_size[2]
	
	data_fetcher = DataFetcher()
	images, classes, _, _ = data_fetcher.get_images(target_size[:2], samples, is_sub_dir, load_by_cv)
	print (images.shape)
	X_train, X_test = train_test_split(images, test_size=0.2, random_state=1)
	X_train = X_train.astype("float32") / 255.0
	X_test = X_test.astype("float32") / 255.0
	(encoder, decoder, autoencoder) = ConvAutoencoder.build(height, width, depth)
	opt = Adam(lr=1e-3)
	autoencoder.compile(loss="mse", optimizer=opt)
	H = autoencoder.fit(X_train, X_train,
						validation_data=(X_test, X_test),
						epochs=epochs,
						batch_size=bs)

	N = np.arange(0, epochs)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_file)

	indices = np.random.choice(X_test.shape[0], samples, replace=False)
	test_samples = X_test[indices]
	decoded = autoencoder.predict(test_samples)
	outputs = None

	for i in range(0, samples):
		original = (test_samples[i] * 255).astype("uint8")
		recon = (decoded[i] * 255).astype("uint8")
		output = np.hstack([original, recon])
		cv2.imwrite(f'./results/{i+1}.png', output)
	return 

if __name__ == '__main__':
	main()
