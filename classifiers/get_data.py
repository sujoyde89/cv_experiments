import os
from os.path import isfile, join
from os import listdir
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
import pandas as pd
import numpy as np
import config

datapath = config.DATAPATH

class DataFetcher:

	def __init__(self):
		pass

	def get_sub_directories(self):
		sub_dirs = next(os.walk(datapath))[1]
		return sub_dirs

	def get_subfile_details(self, dir_path, samples, sub_dir = None):
		file_details = []
		files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))][:samples]
		for file in files:
			file_detail = {}
			file_detail['class'] = sub_dir
			file_detail['path'] = join(dir_path, file) 
			if os.path.getsize(file_detail['path']) > 0:
				file_details.append(file_detail)
		return file_details

	@staticmethod
	def get_image_array(image_path, target_size, load_by_cv):
		if load_by_cv is False:
			if target_size is not None:
				img = load_img(image_path, target_size=target_size)
			else:
				img = load_img(image_path)
			img = img_to_array(img)
		else:
			img = cv2.imread(image_path)
			#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if target_size is not None:
				img = cv2.resize(img, (target_size), interpolation=cv2.INTER_AREA)
		return img

	def image_dataframe(self, df):
		df['size'] = df['image'].apply(lambda x : x.shape)
		#df.to_csv('image_details.csv', index = False)
		return df

	def get_images(self, target_size, samples = 1000, sub_dir=True, load_by_cv = False):
		file_details = []
		if sub_dir:
			sub_dirs = self.get_sub_directories()
			for sub_dir in sub_dirs:
				dir_path = join(datapath, sub_dir)
				file_details.extend(self.get_subfile_details(dir_path, samples, sub_dir))
		else:
			dir_path = datapath
			file_details.extend(self.get_subfile_details(dir_path, samples))

		for file_detail in file_details:
			file_detail['image'] = self.get_image_array(file_detail['path'], target_size=target_size, load_by_cv=load_by_cv)
		df = pd.DataFrame(file_details)
		df = self.image_dataframe(df)
		df = df.sample(frac=1, random_state=12).reset_index(drop=True)
		return np.array([f for f in df['image']]), df['class'].values.tolist(), df['path'].values.tolist(), df

if __name__ == "__main__":
	target_size = config.TARGET_SIZE
	samples = config.SAMPLES
	is_sub_dir = config.IS_SUB_DIR
	load_by_cv = config.LOAD_BY_CV
	data_fetcher = DataFetcher()
	images, classes, names, df = data_fetcher.get_images(target_size[:2], samples, is_sub_dir, load_by_cv)
	print (images.shape)
	print (df['class'].value_counts())
