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


def find_circle(img, model):

    row, col, rad = model.predict(img)

    # Fill in this function
    return row, col, rad


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():


    # load weights into new model
    loaded_model = models.load_model(best_model_path)

    results = []
    for _ in range(1000):

        #Creating the image and preprocessing
        params, img = noisy_circle(size, radius, noise)
        img = preprocess_img(img)
        img = img.reshape((1, size, size, 3))
        #Get the detections
        detected = find_circle(img, loaded_model)

        print("detected: ({}, {}, {}), actual : {}".format(detected[0][0][0], detected[1][0][0], detected[2][0][0], params))

        results.append(iou(params, detected))
    results = np.array(results)
    print("percentage of samples with IOU >0.7 : {} %".format((results > 0.7).mean() * 100))


if __name__ == '__main__':
    main()
