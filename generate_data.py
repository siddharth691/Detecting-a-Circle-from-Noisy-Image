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
from config import *
from helper import *


if not os.path.exists(train_dir):
	os.mkdir(train_dir)

if not os.path.exists(test_dir):
	os.mkdir(test_dir)
	
if not os.path.exists(val_dir):
	os.mkdir(val_dir)
	
#Saving data for training 

files = glob.glob(train_dir + '*')
for f in files:
	os.remove(f)

for i in range(no_train_samples):
	if(i%1000 == 0):
		print("gen train sample: {}".format(i))

	params, img = noisy_circle(size, radius, noise)
	img = preprocess_img(img)
	img_name = train_dir+ "img_"+str(i+1)+".png"
	cv2.imwrite(img_name, img)
	
	with open(train_dir +'labels.csv', 'a+') as f:
		writer = csv.writer(f , lineterminator='\n')
		writer.writerow(params)

print("generated training data")

#Saving data for testing 
files = glob.glob(test_dir + '*')
for f in files:
	os.remove(f)
	
for i in range(no_test_samples):
	params, img = noisy_circle(size, radius, noise)
	img = preprocess_img(img)
	img_name = test_dir + "img_"+str(i+1)+".png"
	cv2.imwrite(img_name, img)	
	with open(test_dir+'labels.csv', 'a+') as f:
		writer = csv.writer(f , lineterminator='\n')
		writer.writerow(params)

print("generated testing data") 
#Saving data for validation 
files = glob.glob(val_dir + '*')
for f in files:
	os.remove(f)

	
for i in range(no_val_samples):
	params, img = noisy_circle(size, radius, noise)
	img = preprocess_img(img)
	img_name = val_dir+ "img_"+str(i+1)+".png"
	cv2.imwrite(img_name, img)
	
	with open(val_dir+'labels.csv', 'a+') as f:
		writer = csv.writer(f , lineterminator='\n')
		writer.writerow(params)


print("generated validation data")
