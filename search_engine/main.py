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
import json
from tqdm import tqdm
from scipy import spatial

from get_data import DataFetcher
from feature_extractor import ColorDescriptor
import config

seed(12)
tf.random.set_seed(12)


def get_similarity(x, test_features, eps=1e-10):
	d = spatial.distance.cosine(x, test_features)
	#d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
    #               for (a, b) in zip(x, test_features)])
	return d
	

def  run(data_exists):

	target_size = config.TARGET_SIZE
	samples = config.SAMPLES
	is_sub_dir = config.IS_SUB_DIR
	load_by_cv = config.LOAD_BY_CV
	limit_similar = config.SIMILAR_LIMIT

	hue_bins = config.HUE_BINS
	sat_bins = config.SAT_BINS
	val_bins = config.VALUE_BINS

	color_descriptor = ColorDescriptor((hue_bins, sat_bins, val_bins))
	feat_cols = ['feat_' + str(x) for x in range(1,(hue_bins * sat_bins * val_bins * 5) + 1)]

	if data_exists:
		image_df = pd.read_csv('image_features.csv')
		
	else:
		data_fetcher = DataFetcher()
		images, _, names, _ = data_fetcher.get_images(target_size = target_size[:2], \
													samples = samples, \
                                                	sub_dir = is_sub_dir, \
													load_by_cv = load_by_cv)
		
		image_details = []
		for img, name in tqdm(zip(images, names)):
			image_dict = {
				'name':name,
				'features':color_descriptor.describe(img),
				#'image':img
			}
			image_details.append(image_dict)

		image_df = pd.DataFrame(image_details)
		image_df_expanded = pd.DataFrame(image_df['features'].tolist(), \
			columns=feat_cols)
		image_df = pd.concat([image_df, image_df_expanded], axis = 1)
		image_df.drop(['features'], axis = 1, inplace = True)
		image_df.to_csv('image_features.csv', index = False)
	
	test_image = '../dataset/jpg/108100.jpg'
	test_image = DataFetcher.get_image_array(test_image, target_size=target_size[:2], load_by_cv=True)
	test_image_features = color_descriptor.describe(test_image)
	image_df['distance'] = image_df[feat_cols].apply(lambda x : get_similarity(x, test_image_features), axis = 1)
	#print (image_df.sort_values(by='similarity', ascending=True).head())
	similar_images = image_df.sort_values(by='distance', ascending=True)[:limit_similar]['name']

	cv2.imwrite(f'./results/image.png', test_image)
	for i, name in enumerate(similar_images):
		print (name)
		img = DataFetcher.get_image_array(name, target_size=target_size[:2], load_by_cv=True)
		cv2.imwrite(f'./results/search_image_{i+1}.png', img)

	return

if __name__ == '__main__':
	print (run(data_exists=True))
