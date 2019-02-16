import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import skimage as ski
import matplotlib.pyplot as plt
import cv2
import csv
import os
import shutil
import glob
import tensorflow as tf

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

from keras.wrappers.scikit_learn import KerasRegressor

from config import *
from helper import *


#Get the testing data
X_test, y_test_row, y_test_col, y_test_rad = get_input_output_data(dir_path = test_dir, no_samples = no_test_samples)


# # load json and create model
# json_file = open(best_model_path, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = models.model_from_json(loaded_model_json)


# # load weights into new model
# loaded_model.load_weights("./model.h5")
# print("Loaded model from disk")

loaded_model = models.load_model(best_model_path)

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='mean_squared_error', loss_weights=loss_weight)
score = loaded_model.evaluate(X_test, [y_test_row, y_test_col, y_test_rad], verbose=1)

print("Metric Name : Scores ")
for i, s in enumerate(score):
	print("{} : {}".format(loaded_model.metrics_names[i], s))

