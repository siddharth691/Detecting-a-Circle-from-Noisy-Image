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

#Helper functions

def preprocess_img(img, noise = noise):
    
    img = img/ (1 + noise*0.999)
    img = np.dstack((img, img, img))
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise, min_radius = 5):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(min_radius, max(min_radius, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


#Read input output data 
def get_input_output_data(dir_path = train_dir, no_samples = no_train_samples):
    y1 = []
    y2 = []
    y3 = []
    with open(dir_path+"labels.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            y1.append(row[0])
            y2.append(row[1])
            y3.append(row[2])
    
    X = []
    for i in range(no_samples):
        #it reads rgb version of this image by default (3 channel) which is the input to the model
        img = cv2.imread(dir_path+"img_"+str(i+1)+'.png')
        X.append(img)

    return np.array(X), y1, y2, y3



#Create keras model with VGG backbone
def vgg_model(vgg_conv, input_size, loss_weight = [1.0, 1.0, 1.0], dropout=dropout):

    Input_image = layers.Input(shape=(size,size,3), name = "Input_Image")
    vgg_out = vgg_conv(Input_image)
    x = layers.Flatten()(vgg_out)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    row_out = layers.Dense(1, kernel_initializer='normal', name = 'row_out')(x)
    col_out = layers.Dense(1, kernel_initializer='normal', name = 'col_out')(x)
    rad_out = layers.Dense(1, kernel_initializer='normal', name = 'rad_out')(x)
    model = models.Model(inputs=Input_image, outputs=[row_out, col_out, rad_out])
    model.compile(optimizer=optimizers.Adam(lr= learning_rate), loss='mean_squared_error', loss_weights=loss_weight)
    
    return model

def step_decay(epoch, lr):
    lrate = lr * 1/(1 + decay_rate * epoch)
    return lrate
