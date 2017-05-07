#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:19:06 2017

@author: hope
"""

import csv
import cv2
import numpy as np

dpath = '/Users/hope/Documents/python/carND/CarND-Behavioral-Cloning-P3/data'

lines = []
with open(dpath + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    csvHeader = next(reader)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    steering_correction = 0.2
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction

                img_center = cv2.imread(dpath + '/IMG/' + batch_sample[0].split('/')[-1])
                img_left   = cv2.imread(dpath + '/IMG/' + batch_sample[1].split('/')[-1])
                img_right  = cv2.imread(dpath + '/IMG/' + batch_sample[2].split('/')[-1])
    
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
                images.extend([cv2.flip(img_center,1),cv2.flip(img_left,1),cv2.flip(img_right,1)])
                angles.extend([steering_center*-1.0, steering_left*-1.0, steering_right*-1.0])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(train_generator, 
                                   samples_per_epoch=len(train_samples)*6, 
                                   validation_data=validation_generator, 
                                   nb_val_samples=len(validation_samples)*6, 
                                   nb_epoch=5)

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

model.save('model_generator.h5')