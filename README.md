#README detect a circle in a noisy image using Convnets


Requirements:
--------------
1. Tensorflow-gpu
2. keras
3. Numpy
4. Skimage
5. Opencv
6. Matplotlib

Directory Structure and Files:
------------------------------
1. Trained_models/Models -- Contains the final trained model
2. data/train - contains the training data (30,000) images and labels.csv file (Not provided to keep the size small)
3. data/test - contains the testing data (4000) images and labels.csv file (Not provided to keep the size small) (two images are given for demo purposes)
4. data/validation - contains the validation data (4000) images and labels.csv file (Not provided to keep the size small)
5. config.py - all the configurable parameters
6. generate_data.py - to generate the data (training, testing and validation), generating parameters such as noise and no of images can be configured in config.py
7. helper.py - all the helper functions
8. train.py - to train and save the model and best checkpoint in the Trained_models folder given different configurable parametes in config.py
9. test.py - to test the best model on the testing dataset and printing out the losses
10. main.py - main file to detect circle loading best model and printing out the accuracy when IOU > 0.7 

Usage:
------
1. To generate data - "python generate_data.py" (set parameters in config.py) - it will save data in data folder
2. To see train - "python train.py" (set the parameters in config.py) - it will save models in Trained_models folder based on best validation scores.
3. To see the test loss - "python test.py" (set the parameters in config.py) - it will print out the losses for testing dataset.
4. To see the detected circle accuracy - "python main.py" (set the parameters in config.py) - it will print out the accuracy.


Tunable parameters/ Best Parameters obtained:
---------------------------------------------
1. To generate all three dataset and train : 
	1. noise = 0.5, on increasing the noise circle was not clearly visible to even human eye so, there was very less learning happening
	2. size = 200 , size of images
	3. radius = 40, max radius of circle 
	4. no_train_samples = 30000
	5. no_test_samples = 4000
	6. no_val_samples = 4000

2. Training parameters:
	1. learning_rate = 0.001 till 40 epochs after that 0.0001 (reloading the checkpoint and epochs starting from 1 to 10)
	2. loss_weight = [1.0, 1.0, 1.0] (for row, col and radius respectively)
	3. NN backbone = VGG16
	4. no_of_layers_trainable = 10 (last 10 layers)
	5. batch_size =20
	6. no_epochs = 10
	7. dropout = 0.3 #Dropout just before the last fully connected layer for outputs
	8. decay_rate = 0.001 (learning rate decreasing with epochs -- learning rate = learning rate (1 + decay_rate * epoch)

Approach:
---------
I have used pretrained VGG and trained last 10 layers for the VGG. After that 3 seperate parallel full connected layers to get 3 losses.
Total loss = 1* loss due to row + 1* loss due to col + 1* loss due to radius

Architecture:
-------------

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input_Image (InputLayer)        (None, 200, 200, 3)  0                                            
__________________________________________________________________________________________________
vgg16 (Model)                   (None, 6, 6, 512)    14714688    Input_Image[0][0]                
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 18432)        0           vgg16[1][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         18875392    flatten_1[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
row_out (Dense)                 (None, 1)            1025        dropout_1[0][0]                  
__________________________________________________________________________________________________
col_out (Dense)                 (None, 1)            1025        dropout_1[0][0]                  
__________________________________________________________________________________________________
rad_out (Dense)                 (None, 1)            1025        dropout_1[0][0]                  
==================================================================================================
Total params: 33,593,155
Trainable params: 32,447,747
Non-trainable params: 1,145,408

Preprocessing of the generated images: scaling between (0,1) -> stacking to make 3 channeled -> normalized -> converting to uint8
Loss used: Mean squared Error
Optimizer: Adam 


Results:
--------
With above parameters:

1. Training Error: total_loss: 164.3115 - row_out_loss: 75.8875 - col_out_loss: 76.7118
2. Validation Error: val_loss: 5.5422 - val_row_out_loss: 1.9598 - val_col_out_loss: 2.0988 - val_rad_out_loss: 1.4835
3. Testing Error: loss : 5.321152105331421 - row_out_loss : 1.5109475388526916 - col_out_loss : 2.2544506106376647 - rad_out_loss : 1.5557539939880372 

4. With noise = 0.5 : detected % of images with IOU > 0.7 = 87.7%  for 1000 images
5. With noise = 0.7 : detected % of images with IOU > 0.7 = 89.5%  for 1000 images
6. With noise = 0.9 : detected % of images with IOU > 0.7 = 88.0%  for 1000 images
7. With noise = 1.1 : detected % of images with IOU > 0.7 = 86.8%  for 1000 images
8. With noise = 1.3 : detected % of images with IOU > 0.7 = 86.2%  for 1000 images
9. With noise = 1.5 : detected % of images with IOU > 0.7 = 86.4%  for 1000 images
10. With noise = 1.7 : detected % of images with IOU > 0.7 = 88%  for 1000 images
11. With noise = 2.0 : detected % of images with IOU > 0.7 = 87.8%  for 1000 images
12. With noise = 3.0 : detected % of images with IOU > 0.7 = 85.2%  for 1000 images
13. With noise = 4.0 : detected % of images with IOU > 0.7 = 86.7%  for 1000 images
14. With noise = 5.0 : detected % of images with IOU > 0.7 = 85.0%  for 1000 images

Although the detector accuracy decreases as we increase the noise in the data as it is trained on less noise, but it has learnt the circle features pretty well
and able to detect the circle with reasonable accuracy.

Further improvements:
---------------------
for better accuracy and low validation loss:
1. Architecture can be experimented
2. can be trained with more data
3. hyper-parameters can be experimented

Device Used:
------------
Tesla P100-PCIE-16GB GPU


