#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import csv
import cv2
import numpy as np

#path to data
dpath = '/Users/hope/Documents/python/carND/CarND-Behavioral-Cloning-P3/data'
"""
Read in driving_log,csv and create the 'lines' list
Each item in the 'lines' list corresponds to a moment in time at which data was 
collected. For each moment, the following data were recorded:
    0. filename of image from center camera
    1. filename of image from left camera
    2. filename of image from right camera
    3. Steering Angle measurement
    4. Throttle measurement
    5. Brake measurement
    6. Speed measurement
"""
lines = []
with open(dpath + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    csvHeader = next(reader)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
"""
split data into training and validation sets - 20% validation 80% training
input data is 'lines' list described above.
"""
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn

"""
Python generator to process large amounts of image data while minimizing 
memory usage.

Divides the training (or validation) data into batches of a specified size.
Processes each batch
For data point in each batch:
    1. read in the steering measurement and paths for the center, left, 
        and right images
    2. read in the center, left, and right images and append to images list
    3. flip the center, left, and right images and append to images list
    4. read in the steering angle measurement
    5. apply a correction factor to the steering measurement that corresponds 
        to  the left and right sides
    6. append the steering angle and corrected steering angles to the 
        angles list
    7. multiply the steering angle measurements from step 6 by -1 and 
        append to angles list
    8 return the images and angles lists
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    #correction to be applied to the steering angle measurement
    #for off-center camera images
    steering_correction = 0.2
    
    while 1: # Loop forever so the generator never terminates
        
    #shuffle the data
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #read in steering measurment and apply corrections for 
                #left and right camera images and append to angles list
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
                angles.extend([steering_center, steering_left, steering_right])

                #read in center, left and right images & append to images list
                img_center = cv2.imread(dpath + '/IMG/' + 
                                        batch_sample[0].split('/')[-1])
                img_left   = cv2.imread(dpath + '/IMG/' + 
                                        batch_sample[1].split('/')[-1])
                img_right  = cv2.imread(dpath + '/IMG/' + 
                                        batch_sample[2].split('/')[-1])
                images.extend([img_center, img_left, img_right])
                
                #flip the images and corresponding steering angles and append
                #to images and angles lists
                images.extend([cv2.flip(img_center,1),cv2.flip(img_left,1),
                               cv2.flip(img_right,1)])
                angles.extend([steering_center*-1.0, steering_left*-1.0, 
                               steering_right*-1.0])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

#set up regression network 
model = Sequential()

"""
preprocess the data
normalize pixel values of each channel to be between -0.5 and 0.5
crop the top 70 pixels and lower 25 pixels from each image 
"""
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

"""
Architecture based on the NVIDIA architecture presented in the lectures
5 convolutional layers followed by 4 fully connected layers 
Output is a predicted steering measurement
"""
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#use mean square error loss function 
#minimize error between predicted steering measurement and actual measurement
model.compile(loss='mse', optimizer='adam')

history_object=model.fit_generator(train_generator, 
                                   samples_per_epoch=len(train_samples)*6, 
                                   validation_data=validation_generator, 
                                   nb_val_samples=len(validation_samples)*6, 
                                   nb_epoch=10)

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save the model
model.save('model.h5')