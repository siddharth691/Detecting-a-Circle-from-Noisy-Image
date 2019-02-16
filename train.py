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
from keras import callbacks

from keras.wrappers.scikit_learn import KerasRegressor

from config import *
from helper import *

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))

# Freeze the layers except the last "no_of_layers_trainable" layers
for layer in vgg_conv.layers[:-1*no_of_layers_trainable]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
# Show a summary of the model. Check the number of trainable parameters
model = vgg_model(vgg_conv, size, loss_weight = loss_weight)
model.summary()

#Get the training data to fine tune the vgg model
X_train, y_train_row, y_train_col, y_train_rad = get_input_output_data()

#Get the validation data
X_val, y_val_row, y_val_col, y_val_rad = get_input_output_data(dir_path = val_dir, no_samples = no_val_samples)

#Get the testing data
X_test, y_test_row, y_test_col, y_test_rad = get_input_output_data(dir_path = test_dir, no_samples = no_test_samples)

# checkpoint saving
filepath=exp_path + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
lrate = callbacks.LearningRateScheduler(step_decay)

callbacks_list = [checkpoint, lrate]


if(load_check_point == True):
	model.load_weights(load_check_point_path)


#Fine tuning the model
model.fit(X_train, [y_train_row, y_train_col, y_train_rad], batch_size=batch_size, epochs=no_epochs, verbose=1,\
          validation_data = (X_val, [y_val_row, y_val_col, y_val_rad]),callbacks=callbacks_list)


# serialize model to JSON
model_json = model.to_json()
with open("./model.json", "w+") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
