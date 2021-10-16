import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 as cv
import glob
import numpy as np
import zipfile

from numpy.lib.type_check import imag
from process_image import *
import shutil
import sys
import subprocess
from threading import Thread
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import gc


_translate = QtCore.QCoreApplication.translate

BACK_GROUND_COLOR = (255, 255, 255)
"""def process_image(image, min_x, max_x, min_y, max_y, red_min, red_max, green_min, green_max, blue_min, blue_max, progress_label, progress_bar, app, force_redraw, mask):

	_translate = QtCore.QCoreApplication.translate
	progress_label.setText(_translate("MainWindow", "Cropping the image ..."))
	app.processEvents()

	processed_image_np = np.zeros(image.shape)
	processed_mask_np = np.zeros(mask.shape)

	if min_x < max_x and min_y < max_y:
		print("First is true")
		processed_image_np[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]
		processed_mask_np[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
	
	elif min_x > max_x and min_y < max_y:
		print("Second is true")
		processed_image_np[min_y:max_y, :max_x] = image[min_y:max_y, :max_x]
		processed_image_np[min_y:max_y, min_x:] = image[min_y:max_y, min_x:]

		processed_mask_np[min_y:max_y, :max_x] = mask[min_y:max_y, :max_x]
		processed_mask_np[min_y:max_y, min_x:] = mask[min_y:max_y, min_x:]
	
	elif min_x < max_x and min_y > max_y:
		print("Third is true")
		processed_image_np[:max_y, min_x:max_x] = image[:max_y, min_x:max_x]
		processed_image_np[min_y:, min_x:max_x] = image[min_y:, min_x:max_x]

		processed_mask_np[:max_y, min_x:max_x] = mask[:max_y, min_x:max_x]
		processed_mask_np[min_y:, min_x:max_x] = mask[min_y:, min_x:max_x]
	
	else:
		print("Fourth is true")
		processed_image_np[:, :max_x] = image[:, :max_x]
		processed_image_np[:, min_x:] = image[:, min_x:]

		processed_image_np[:max_y, :] = image[:max_y, :]
		processed_image_np[min_y:, :] = image[min_y:, :]

		processed_mask_np[:, :max_x] = mask[:, :max_x]
		processed_mask_np[:, min_x:] = mask[:, min_x:]

		processed_mask_np[:max_y, :] = mask[:max_y, :]
		processed_mask_np[min_y:, :] = mask[min_y:, :]
	
	progress_bar.setValue(10)
	progress_label.setText(_translate("MainWindow", "Applying Threshold ..."))
	one_percent = int(processed_image_np.shape[0]/90.0)
	app.processEvents()

	if red_min < red_max:
		if green_min < green_max:
			if blue_min < blue_max:
				print(1)
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max) or (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max) or (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max):
							processed_image_np[i][j] = (0, 0, 0)

					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
			else:
				print(2)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max) or (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max) or (not (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max)):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
		else:
			temp = green_min
			green_min = green_max
			green_max = temp
			if blue_min < blue_max:
				print(3)
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max) or (not (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max)) or (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
			else:
				print(4)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max) or (not (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max)) or (not (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max)):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
	else:
		temp = red_min
		red_min = red_max
		red_max = temp
		if green_min < green_max:
			if blue_min < blue_max:
				print(5)
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (not (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max)) or (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max) or (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
			else:
				print(6)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (not (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max)) or (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max) or (not (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max)):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
		else:
			temp = green_min
			green_min = green_max
			green_max = temp
			if blue_min < blue_max:
				print(7)
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (not (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max)) or (not (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max)) or (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max):
							processed_image_np[i][j] = (0, 0, 0)

					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
			
			else:
				print(8)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				for i in range(processed_image_np.shape[0]):
					for j in range(processed_image_np.shape[1]):
						if (not (processed_mask_np[i][j][2] < red_min or processed_mask_np[i][j][2] > red_max)) or (not (processed_mask_np[i][j][1] < green_min or processed_mask_np[i][j][1] > green_max)) or (not (processed_mask_np[i][j][0] < blue_min or processed_mask_np[i][j][0] > blue_max)):
							processed_image_np[i][j] = (0, 0, 0)
					if i % 50 == 0:
						progress_bar.setValue(10+int(i/one_percent))
						if force_redraw:
							app.processEvents()
    
	processed_image = processed_image_np.astype(np.uint8)

	return processed_image"""

def apply_white_background(img):
	white_image = np.ones(img.shape)*255
	white_image = white_image.astype(np.uint8)
	frame1 = cv.inRange(img, (0, 0, 0), (0, 0, 0))
	#frame1 = cv.bitwise_not(frame1)
	frame = cv.bitwise_and(white_image, white_image, mask=frame1)
	'''cv.imshow("White Area", frame)
	cv.waitKey(0)
	cv.destroyAllWindows()'''
	print("I made the background white again")
	img += frame
	return img

def process_image(image, min_x, max_x, min_y, max_y, red_min, red_max, green_min, green_max, blue_min, blue_max, progress_label, progress_bar, app, force_redraw, mask, apply_white_back=True):

	progress_label.setText(_translate("MainWindow", "Cropping the image ..."))
	app.processEvents()

	processed_image_np = np.zeros(image.shape)
	processed_mask_np = np.zeros(mask.shape)

	if min_x < max_x and min_y < max_y:
		print("First is true")
		processed_image_np[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]
		processed_mask_np[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
	
	elif min_x > max_x and min_y < max_y:
		print("Second is true")
		processed_image_np[min_y:max_y, :max_x] = image[min_y:max_y, :max_x]
		processed_image_np[min_y:max_y, min_x:] = image[min_y:max_y, min_x:]

		processed_mask_np[min_y:max_y, :max_x] = mask[min_y:max_y, :max_x]
		processed_mask_np[min_y:max_y, min_x:] = mask[min_y:max_y, min_x:]
	
	elif min_x < max_x and min_y > max_y:
		print("Third is true")
		processed_image_np[:max_y, min_x:max_x] = image[:max_y, min_x:max_x]
		processed_image_np[min_y:, min_x:max_x] = image[min_y:, min_x:max_x]

		processed_mask_np[:max_y, min_x:max_x] = mask[:max_y, min_x:max_x]
		processed_mask_np[min_y:, min_x:max_x] = mask[min_y:, min_x:max_x]
	
	else:
		print("Fourth is true")
		processed_image_np[:, :max_x] = image[:, :max_x]
		processed_image_np[:, min_x:] = image[:, min_x:]

		processed_image_np[:max_y, :] = image[:max_y, :]
		processed_image_np[min_y:, :] = image[min_y:, :]

		processed_mask_np[:, :max_x] = mask[:, :max_x]
		processed_mask_np[:, min_x:] = mask[:, min_x:]

		processed_mask_np[:max_y, :] = mask[:max_y, :]
		processed_mask_np[min_y:, :] = mask[min_y:, :]
	
	progress_bar.setValue(10)
	progress_label.setText(_translate("MainWindow", "Applying Threshold ..."))
	one_percent = int(processed_image_np.shape[0]/90.0)
	app.processEvents()
	processed_image_np = processed_image_np.astype(np.uint8)
	processed_mask_np = processed_mask_np.astype(np.uint8)

	if red_min < red_max:
		if green_min < green_max:
			if blue_min < blue_max:
				print(1)
				frame_threshold = cv.inRange(processed_mask_np, (blue_min, green_min, red_min), (blue_max, green_max, red_max))
				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
	
			else:
				print(2)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				
				frame1 = cv.inRange(processed_mask_np, (0, green_min, red_min), (blue_min, green_max, red_max))
				frame2 = cv.inRange(processed_mask_np, (blue_max, green_min, red_min), (255, green_max, red_max))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
				
			
		else:
			temp = green_min
			green_min = green_max
			green_max = temp
			if blue_min < blue_max:
				print(3)
				frame1 = cv.inRange(processed_mask_np, (blue_min, 0, red_min), (blue_max, green_min, red_max))
				frame2 = cv.inRange(processed_mask_np, (blue_min, green_max, red_min), (blue_max, 255, red_max))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
			
			else:
				print(4)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp

				frame1 = cv.inRange(processed_mask_np, (0, 0, red_min), (blue_min, green_min, red_max))
				frame2 = cv.inRange(processed_mask_np, (blue_max, green_max, red_min), (255, 255, red_max))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
	else:
		temp = red_min
		red_min = red_max
		red_max = temp
		if green_min < green_max:
			if blue_min < blue_max:
				print(5)
				frame1 = cv.inRange(processed_mask_np, (blue_min, green_min, 0), (blue_max, green_max, red_min))
				frame2 = cv.inRange(processed_mask_np, (blue_min, green_min, red_max), (blue_max, green_max, 255))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
			
			else:
				print(6)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp

				frame1 = cv.inRange(processed_mask_np, (0, green_min, 0), (blue_min, green_max, red_min))
				frame2 = cv.inRange(processed_mask_np, (blue_max, green_min, red_max), (255, green_max, 255))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
			
		else:
			temp = green_min
			green_min = green_max
			green_max = temp
			if blue_min < blue_max:
				print(7)
				
				frame1 = cv.inRange(processed_mask_np, (blue_min, 0, 0), (blue_max, green_min, red_min))
				frame2 = cv.inRange(processed_mask_np, (blue_min, green_max, red_max), (blue_max, 255, 255))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
			
			else:
				print(8)
				temp = blue_min
				blue_min = blue_max
				blue_max = temp
				frame1 = cv.inRange(processed_mask_np, (0, 0, 0), (blue_min, green_min, red_min))
				frame2 = cv.inRange(processed_mask_np, (blue_max, green_max, red_max), (255, 255, 255))

				frame_threshold = cv.bitwise_or(frame1, frame2)

				processed_image_np = cv.bitwise_and(processed_image_np, processed_image_np, mask=frame_threshold)
    

	if apply_white_back:
		processed_image_np = apply_white_background(processed_image_np)

	processed_image = processed_image_np.astype(np.uint8)

	

	return processed_image


def get_best_ii_jj(num):

	ii = 0
	jj = 0

	if num >= 25:
		ii = 5
		jj = 5

	elif int(num**0.5) == num**0.5:
		ii = int(num**0.5)
		jj = ii
	
	else:
		ii = int(num**0.5)+1
		jj = int(num/ii)+1
	
	return ii, jj

def put_all_images_together(images, dir):

	ii, jj = get_best_ii_jj(images.shape[0])

	print((ii, jj))
	if ii > 1 and jj > 1:	
		fig, axs = plt.subplots(ii, jj)

		for i in range(ii):
			for j in range(jj):
				if images.shape[0] > (i*ii)+j:
					axs[i, j].imshow(images[(i*ii)+j])
				else:
					break
	else:
		plt.imshow(images[0])
		
	img_path = dir + '/fig.png'
	
	plt.savefig(img_path)

	return img_path

def use_kmeans(images_pathes, progress_bar, progress_label, app, dir, use_original_size= False):

	progress_label.setText(_translate("MainWindow", "Processing images for KMeans clustering ..."))
	progress_bar.setValue(1)
	app.processEvents()

	images = []
	img = cv.imread(images_pathes[0])
	ratio = img.shape[0]/img.shape[1]

	for path in images_pathes:
		try:
			img = cv.imread(path)
			if not use_original_size:
				img = cv.resize(img, (int(100), int(ratio*100)) )
			images.append(img)
		except:
			print("Image not found or th file is not an image")
	
	progress_label.setText(_translate("MainWindow", "Calculating the best number of clusters for KMeans ..."))
	progress_bar.setValue(2)
	app.processEvents()

	training_images = np.array(images)
	sample_size = 5
	if training_images.shape[0] < 5:
		sample_size = training_images.shape[0]

	random_sample = np.random.randint(0,training_images.shape[0],sample_size)
	sample_images = []
	for i in random_sample:
		sample_images.append(images[i])

	images = np.array(images)
	sample_images = np.array(sample_images)

	### find best number of clusters
	silhouette_scores = []
	print(sample_images.shape)
	
	shape_1 = sample_images.shape[0]*sample_images.shape[1]*sample_images.shape[2]
	x = sample_images.reshape(shape_1, sample_images.shape[3])
	kmeanses = []

	for k in range(2, 11, 1):
		kmeans = KMeans(n_clusters=k)
		kmeans.fit(x)
		score = silhouette_score(x, kmeans.labels_)
		if k >= 4:
			if score < silhouette_scores[len(silhouette_scores) - 1] and score < silhouette_scores[len(silhouette_scores) - 2]:
				break
		silhouette_scores.append(score)
		kmeanses.append(kmeans)
		progress_bar.setValue(int(k * (90/(10-2))))
		app.processEvents()
	
	progress_bar.setValue(90)
	app.processEvents()

	best_k = silhouette_scores.index(max(silhouette_scores)) + 2

	best_kmeans = kmeanses[silhouette_scores.index(max(silhouette_scores))]


	progress_label.setText(_translate("MainWindow", "Applying to 25 images (" + str(best_k) + " clusters)..."))

	### Clustering using the best k found
	
	num = 25
	if training_images.shape[0] < 25:
		num = training_images.shape[0]
	percentage = num/9

	for i in range(num):
		img = training_images[i]
		shape_1 = img.shape[0]*img.shape[1]
		x = img.reshape(shape_1, img.shape[2])
		labels = best_kmeans.predict(x)
		images[i] = (best_kmeans.cluster_centers_[labels]).reshape((images[i].shape[0], images[i].shape[1], images[i].shape[2]))

		progress_bar.setValue(90 + int(i/percentage))
		app.processEvents()

	image_path = put_all_images_together(images, dir)

	progress_label.setText(_translate("MainWindow", "Done, click apply and then save to apply this change to the original images and save changes..."))
	progress_bar.setValue(100)
	app.processEvents()

	return image_path, best_kmeans, best_k
'''
def prediction_to_filter(prediction, img_shape, window_size):

	test_filter = np.zeros((img_shape[0], img_shape[1]))

	for i in range(window_size[0]):

		for j in range(window_size[1]):
		
			h_index = np.arange(window_size[1] - j, img_shape[1], window_size[1])
			v_index = np.arange(window_size[0] - i, img_shape[0], window_size[0])

			windows_1 = np.vsplit(test_filter, v_index)

			for k in range(len(windows_1)):
				imgy = windows_1[k]

				imgy = np.hsplit(imgy, h_index)

				for l in range(len(imgy)):
					w = imgy[l]

					label = prediction[0]
					prediction = np.delete(prediction, 0, axis=0)
					
					v = v_index[k - 1]
					c = h_index[l - 1]
					test_filter[v][c] = label
						
	
	test_filter = test_filter.astype(np.uint8)
	test_filter = test_filter*255

	return test_filter'''

def prediction_to_filter(prediction, img_shape):

	image_filter = prediction.reshape((img_shape[0], img_shape[1]))

	image_filter = image_filter.astype(np.uint8)

	image_filter = image_filter*255

	return image_filter


"""def get_data_from_images(img_pathes, masks_pathes, window_size, progress_bar, app, val, dir, use_original_size= False, model = None):

	data = []
	labels = []
	
	percent = 25.0/len(img_pathes)
	count = 1
	for p in range(len(img_pathes)):
		path = img_pathes[p]
		mask = None
		if masks_pathes is not None:
			mask_path = masks_pathes[p]
			mask = cv.imread(mask_path)
		
		img = cv.imread(path)
		
		
		if model is not None:
			data = []
			labels = []

		if not use_original_size:
			ratio = img.shape[0]/img.shape[1]
			basic_dim = 1000
			if model is not None:
				basic_dim = 400
			img = cv.resize(img, (int(basic_dim), int(ratio*basic_dim)))
			if mask is not None:
				mask = cv.resize(mask, (int(basic_dim), int(ratio*basic_dim)))
		
		test_filter = np.zeros((img.shape[0], img.shape[1]))

		for i in range(window_size[0]):

			for j in range(window_size[1]):
				
				v_index = np.arange(window_size[0] - i, img.shape[0], window_size[0])
				h_index = np.arange(window_size[1] - j, img.shape[1], window_size[1])

				windows_1 = np.vsplit(img, v_index)
				if mask is not None:
					windows_1_mask = np.vsplit(mask, v_index)

				for k in range(len(windows_1)):
					imgy = windows_1[k]
					imgy = np.hsplit(imgy, h_index)

					if mask is not None:
						masky = windows_1_mask[k]
						masky = np.hsplit(masky, h_index)

					for l in range(len(imgy)):
						w = imgy[l]
						w_mask = None
						if mask is not None:
							w_mask = masky[l]

						if w.shape[0] != window_size[1] or w.shape[1] != window_size[0]:

							while w.shape[0] < window_size[1]:
								new_row = np.ones((1, w.shape[1], 3))
								new_row = new_row*255
								new_row = new_row.astype(np.uint8)
								w = np.append(w, new_row, axis=0)
								if w_mask is not None:
									w_mask = np.append(w_mask, new_row, axis=0)

							while w.shape[1] < window_size[0]:
								new_col = np.ones((w.shape[0], 1, 3))
								new_col = new_col*255
								new_col = new_col.astype(np.uint8)
								w = np.append(w, new_col, axis=1)
								if w_mask is not None:
									w_mask = np.append(w_mask, new_col, axis=1)
						
						label = 1.0
						if w_mask is not None:
							value = w_mask[int(window_size[1] / 2)][int(window_size[0] / 2)]

							if value[0] == BACK_GROUND_COLOR[0] and value[1] == BACK_GROUND_COLOR[1] and value[2] == BACK_GROUND_COLOR[2]:
								label = 0.0
						
						v = v_index[k - 1]
						c = h_index[l - 1]
						test_filter[v][c] = label
						
						#w = w.reshape(-1)
						w = w/255.0
						data.append(w)
						labels.append(label)
		
		'''image_filter = prediction_to_filter(labels, img.shape, window_size)
		#image_filter = test_filter.astype(np.uint8)
		#image_filter = image_filter*255
		imgyyy = cv.bitwise_and(img, img, mask = image_filter)
		cv.imshow(path, imgyyy)
		cv.waitKey(0)
		cv.destroyAllWindows()'''


		if model is not None:
			data = np.array(data)
			prediction = model.predict_classes(data, batch_size=1000, verbose=1)

			image_filter = prediction_to_filter(prediction, img.shape, window_size)

			img = cv.bitwise_and(img, img, mask = image_filter)

			name = path.split('/')
			name = name[len(name)-1]

			save_path = dir + "/" + name
			img = apply_white_background(img)
			cv.imwrite(save_path, img)
			pathes.append(save_path)

			'''image = cv.resize(img, (800, 600))
			cv.imshow(path, image)
			cv.waitKey(0)
			cv.destroyAllWindows()'''


		
		percentage = val + count*percent
		progress_bar.setValue(int(percentage))
		print(percentage)
		app.processEvents()
		count += 1
		
	data = np.array(data)
	
	labels = np.array(labels)

	if model == None:
		return data, labels
	else:
		return True
"""
def slide_window(img_pathes, masks_pathes, window_size, progress_bar, app, start_val, end_val, dir, pathes_list, use_original_size= False, model = None):
	data = []
	labels = []
	percent = 1
	if len(img_pathes) > 0:
		percent = (end_val - start_val)/len(img_pathes)
	count = 1

	for p in range(len(img_pathes)):
		path = img_pathes[p]
		mask = None
		if masks_pathes is not None:
			mask_path = masks_pathes[p]
			mask = cv.imread(mask_path)
		
		img = cv.imread(path)
		
		if model is not None:
			data = []
			labels = []

		if not use_original_size:
			ratio = img.shape[0]/img.shape[1]
			basic_dim = 1000
			if model is not None:
				basic_dim = 400
			img = cv.resize(img, (int(basic_dim), int(ratio*basic_dim)))
			if mask is not None:
				mask = cv.resize(mask, (int(basic_dim), int(ratio*basic_dim)))
		
		img_shape = img.shape

		padding_size_0 = int(window_size[0]/2.0) + 1
		padding_size_1 = int(window_size[1]/2.0) + 1

		img_pad = cv.copyMakeBorder(img, padding_size_1, padding_size_1, padding_size_0, padding_size_0, cv.BORDER_CONSTANT, None, BACK_GROUND_COLOR)

		if mask is not None:
			mask_pad = cv.copyMakeBorder(mask, padding_size_1, padding_size_1, padding_size_0, padding_size_0, cv.BORDER_CONSTANT, None, BACK_GROUND_COLOR)

		start_index_i = padding_size_0 - 1
		start_index_j = padding_size_1 - 1

		for i in range(start_index_i, img_shape[0] + start_index_i):
			for j in range(start_index_j, img_shape[1] + start_index_j):
				y_1 = i - int(window_size[0]/2.0)
				y_2 = i + int(window_size[0]/2.0) + 1
				x_1 = j - int(window_size[1]/2.0)
				x_2 = j + int(window_size[1]/2.0) + 1

				w = img_pad[y_1:y_2, x_1:x_2]
				label = 1

				if mask is not None:
					win = mask_pad[y_1:y_2, x_1:x_2]
					v = int(window_size[0]/2.0)
					c = int(window_size[1]/2.0)
					value = win[v][c]

					if value[0] == BACK_GROUND_COLOR[0] and value[1] == BACK_GROUND_COLOR[1] and value[2] == BACK_GROUND_COLOR[2]:
						label = 0
				
				w = w/255.0
				data.append(w)
				labels.append(label)

		if model is not None:
			data = np.array(data, dtype=np.float32)

			bat = 1000

			if pathes_list is not None:
				bat = 10000

			prediction = model.predict_classes(data, batch_size=bat, verbose=1)

			data = None

			image_filter = prediction_to_filter(prediction, img.shape)

			img = cv.bitwise_and(img, img, mask = image_filter)

			name = path.split('/')
			name = name[len(name)-1]

			save_path = dir + "/" + name
			img = apply_white_background(img)
			cv.imwrite(save_path, img)
			img = None
			mask = None
			image_filter = None
			
			if pathes_list is not None:
				pathes_list.append(save_path)
		
		percentage = start_val+ count*percent
		progress_bar.setValue(int(percentage))
		print(percentage)
		app.processEvents()
		count += 1

		gc.collect()

	data = np.array(data, dtype=np.float32)
	
	labels = np.array(labels)

	if model == None:
		return data, labels
	else:
		return True


def use_deep_learning(progressed_images_pathes, original_images_pathes, to_process_images_pathes, progress_bar, progress_label, app, dir, n_jobs = 4, window_size = (5, 5), use_original_size= False):
	
	PATHES = []

	progress_label.setText(_translate("MainWindow", "Preprocessing data for training ..."))
	app.processEvents()

	data, labels = slide_window(original_images_pathes, progressed_images_pathes, window_size, progress_bar, app, 0, 25, dir, None, use_original_size=False)
	
	x_train, x_validation, y_train, y_validation = train_test_split(data, labels, test_size = 0.15)

	progress_label.setText(_translate("MainWindow", "Preparing deep learning model ..."))
	progress_bar.setValue(25)
	app.processEvents()

	model = keras.Sequential()
	model.add(keras.layers.Conv2D(20, (3, 3), input_shape = (window_size[0], window_size[1], 3), activation= "tanh"))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(1024, activation= "tanh"))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(256, activation= "relu"))
	model.add(keras.layers.Dense(2, activation= "sigmoid"))

	print(model.summary())

	progress_label.setText(_translate("MainWindow", "Training the model ..."))
	app.processEvents()
	model.compile (keras.optimizers.Adamax(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999), loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
	model.fit(x_train, y_train, batch_size=1000, steps_per_epoch=int(x_train.shape[0]/1000), epochs=1, validation_data= (x_validation, y_validation), shuffle= 1)
	progress_bar.setValue(37)
	app.processEvents()
	model.fit(x_train, y_train, batch_size=1000, steps_per_epoch=int(x_train.shape[0]/1000), epochs=1, validation_data= (x_validation, y_validation), shuffle= 1)
	progress_bar.setValue(50)
	app.processEvents()

	data = None
	labels = None

	gc.collect()
	
	progress_label.setText(_translate("MainWindow", "Preprocessing data for segmentation ..."))
	app.processEvents()

	pathes_list = []
	data, model = slide_window(to_process_images_pathes, None, window_size, progress_bar, app, 50, 90, dir, pathes_list, use_original_size, model)
	
	"""i = 0
	part = int(len(to_process_images_pathes)/n_jobs)
	threads = []

	
	while i < len(to_process_images_pathes):
		if len(threads) <= n_jobs:
			args_list = [to_process_images_pathes[i:i+part], None, window_size, progress_bar, app, 50, 90, dir, pathes_list, use_original_size, model]
			t = Thread(target=slide_window, args=args_list)
			t.start()
			threads.append(t)
			i += part
		if len(threads) == n_jobs:
			for t in threads:
				t.join()
	"""

	#pathes = get_data_from_images(to_process_images_pathes, to_process_images_pathes, window_size, progress_bar, app, 50, dir, use_original_size, model=model)

	progress_label.setText(_translate("MainWindow", "Segmentation in progress ..."))
	app.processEvents()

	progress_label.setText(_translate("MainWindow", "Done"))

	progress_bar.setValue(100)
	app.processEvents()

	return pathes_list, model
	




