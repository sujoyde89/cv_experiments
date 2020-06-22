import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import seed
import tensorflow as tf
import random

from get_data import DataFetcher
from classifiers.lenet import LeNet
from classifiers.efficient_net import EfficientNetB
import config

model_weight_path = config.MODEL_WEIGHT_PATH
epochs = config.EPOCHS
bs = config.BS
plot_file = config.HISTORY_FILE
target_size = config.TARGET_SIZE
samples = config.SAMPLES
is_sub_dir = config.IS_SUB_DIR
load_by_cv = config.LOAD_BY_CV

seed(12)
random.seed(12)
tf.random.set_seed(12)

label_encoder = LabelEncoder()


def train_model(model, opt, X_train, y_train_labels, X_test, y_test_labels):

	print (model.name)

	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	H = model.fit(X_train, y_train_labels, validation_data=(X_test, y_test_labels), \
					epochs=epochs, batch_size=bs)

	model.save_weights(f'{model_weight_path}/{model.name}.h5')

	N = np.arange(0, epochs)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["accuracy"], label="train_accuracy")
	plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(f'{model_weight_path}/{model.name}.png')

	return H
	
def  main():

	height = target_size[0]
	width = target_size[1]
	depth = target_size[2]

	data_fetcher = DataFetcher()
	images, classes, _, _ = data_fetcher.get_images(target_size[:2], samples, is_sub_dir, load_by_cv)
	vec_classes = label_encoder.fit_transform(classes)
	num_classes = pd.Series(vec_classes).nunique()
	print (images.shape)
	print (num_classes)

	X_train, X_test, y_train, y_test = train_test_split(images, vec_classes, test_size=0.2, random_state=1)
	X_train = X_train.astype("float32") / 255.0
	X_test = X_test.astype("float32") / 255.0

	y_train_labels = utils.to_categorical(y_train, num_classes)
	y_test_labels = utils.to_categorical(y_test, num_classes)
	
	
	opt = Adam(lr=1e-3)

	lenet = LeNet.build(height, width, depth, num_classes)
	train_model(lenet, opt, X_train, y_train_labels, X_test, y_test_labels)

	eff_net = EfficientNetB.build(height, width, depth, num_classes)
	train_model(eff_net, opt, X_train, y_train_labels, X_test, y_test_labels)

	return 

if __name__ == '__main__':
	main()
