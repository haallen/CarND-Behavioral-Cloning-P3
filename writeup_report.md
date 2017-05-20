#**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./overfitting.png "Overfitting"
[image2]: ./loss.png "Final loss function"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 a video of my car driving a lap around the track autonomously

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the NVIDIA architecture presented in the lecture videos. It consists of 5 convolutional layers followed by 4 fully connected layers. The 5 convolutional layers used varying filter sizes and depths as well as strides. 

I preprocessed the data before running it through the network by first normalizing the pixel values to be between -0.5 and 0.5 and also by cropping the top and bottom of each image. 

####2. Attempts to reduce overfitting in the model
I split the data into training and validation sets. I used the training set to train the model and the validation set to determine if the model was over or under fitting. I first ran my model for 10 epochs and noticed that the validation loss decreased for the first 6 epochs but then started increasing. I decreased the number of epochs to 6 and then retrained my model. See the images at the end of this report for graphs of the loss functions for these 2 cases.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

####4. Appropriate training data

I used all of the provided images (including left and right images) to train my network (after splitting 20% off for validation). I also flipped each of the left, right, and center images to augment the training data and help my model generalize to all road conditions.

Note: I did not have a joystick (or even a mouse), so recording my own training data did not seem like it would be beneficial. Plus my network did a pretty good job without the need to create more data.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the approach described in the lecture videos. 

My first step was to use a convolution neural network model similar to the Lenet-5 model from previous lessons (minus the max-pooling). I thought this model might be appropriate because it provides a way inputting images from the simulator into the network and outputting a predicted steering value based on those images. In previous lessons, the Lenet-5 architecture did a great job with classifying traffic signs, so I figured it would be a good place to start for this project.

I used a mean square error loss function and the adam optimizer to minimize the error between my network's predicted steering measurement and the actual measurement during training.

I also preprocessed the data by normalizing the pixel values of each channel of each image to be between -0.5 and 0.5. I then cropped the top 70 pixels and lower 25 pixels from each image because they do not provide any relevant information to the steering angle prediction problem.

To provide more training data and to prevent the model from only being able to drive in counter-clockwise circles, I flipped the image from the center camera and the associated steering angle measurement, and appended them to the data set, after preprocessing the images as described above.

In order to gauge how well the model was working, I shuffled the data and split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I determined this was due to lack of training data. 

In addition to the images from the center camera, I supplemented my training data by incorporating the images from the left and right cameras. I did this by correcting the steering angle measurement by a fixed value (+0.2/-0.2 for the left and right images, respectively). I also flipped these images and added them to the training data as described above. 

All of this new data was also preprocessed, shuffled, and split into training/validation as described above. My laptop couldn't handle this amount of data in memory all at once, so I used a generator to process the data in batches.

Despite all of these changes, my car did a pretty good job of staying on the road except in a few cases. I decided to tweak my overall architecture (see below).

####2. Final Model Architecture

The final model architecture (model.py lines 137-146) is based on the NVIDIA architecture described in the lecture videos. It is 5 convolutional layers followed by 4 fully connected layers. My car then went around the entire track without issue!

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 65x320x3 color image (cropped from original 160x320x3) | 
| Convolution 5x5x24     	| 2x2 stride | RELU activation 	|					 
| Convolution 5x5x36     	| 2x2 stride | RELU activation 	|
| Convolution 5x5x48     	| 2x2 stride | RELU activation 	|
| Convolution 3x3x64     	| 1x1 stride | RELU activation 	|
| Convolution 3x3x64     	| 1x1 stride | RELU activation 	|
| Fully connected		      | output = 100        					|
| Fully connected		      | output = 50         					|
| Fully connected		      | output = 10         					|
| Fully connected		      | output = 1         					  |

####3. Creation of the Training Set & Training Process

I used all of the data that was provided to me (left, right, and center images). A discussion about how I preprocessed, augmented, and divided the data into training and validation sets is described above.

I used the training set to train the model and the validation set to determine if the model was over or under fitting. I first ran my model for 10 epochs and noticed that the validation loss decreased for the first 6 epochs but then started increasing. I decreased the number of epochs to 6 and then retrained my model. See the images below for graphs of the loss functions for these 2 cases.

![alt text][image1]
![alt text][image2]

