#!/usr/bin/env python2

"""
Created on Sun Feb  4 16:41:33 2018

@author: ashishbasireddy
"""

import csv
import codecs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

lines = []
with open('/Users/ashishbasireddy/CarND-Behavioral-Cloning-P3/data/combined_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        print(line)
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

def generator(lines, batch_size=512):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        #shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            measuremants = []
            for line in batch_samples:
                for i in range(3):        
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/Users/ashishbasireddy/CarND-Behavioral-Cloning-P3/data/IMG/' + filename 
                    image = cv2.imread(current_path)
                    measuremant = float(line[3])
                    if (i == 1) and (abs(measuremant) > 0.1):
                        print(current_path)
                        measuremant += 0.2
                        images.append(image)
                        measuremants.append(measuremant)
                        images.append(cv2.flip(image,1))
                        measuremants.append(measuremant*-1.0)
                    if (i == 2) and (abs(measuremant) > 0.1):
                        print(current_path)
                        measuremant -= 0.2
                        images.append(image)
                        measuremants.append(measuremant)
                        images.append(cv2.flip(image,1))
                        measuremants.append(measuremant*-1.0)
                    if (i == 0):
                        print(current_path)
                        images.append(image)
                        measuremants.append(measuremant)
                        images.append(cv2.flip(image,1))
                        measuremants.append(measuremant*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measuremants)
            yield sklearn.utils.shuffle(X_train, y_train)

#creating features and labels.

#for line in lines[1:]:
#    for i in range(3):        
#        source_path = line[i]
#        filename = source_path.split('/')[-1]
#        current_path = '/Users/ashishbasireddy/CarND-Behavioral-Cloning-P3/data/IMG/' + filename 
#        image = cv2.imread(current_path)
#        measuremant = float(line[3])
#        if (i == 1) and (abs(measuremant) > 0.1):
#            print(current_path)
#            measuremant += 0.2
#            images.append(image)
#            measuremants.append(measuremant)
#            images.append(cv2.flip(image,1))
#            measuremants.append(measuremant*-1.0)
#        if (i == 2) and (abs(measuremant) > 0.1):
#            print(current_path)
#            measuremant -= 0.2
#            images.append(image)
#            measuremants.append(measuremant)
#            images.append(cv2.flip(image,1))
#            measuremants.append(measuremant*-1.0)
#        if (i == 0):
#            print(current_path)
#            images.append(image)
#            measuremants.append(measuremant)
#            images.append(cv2.flip(image,1))
#            measuremants.append(measuremant*-1.0)
 

train_generator = generator(train_samples, batch_size=512)
validation_generator = generator(validation_samples, batch_size=512)
        
#augmented data
#augmented_images, augmented_measuremants=images, measuremants
#for image,measuremant in zip(images, measuremants):
#    if abs(measuremant) > 0.1:
#        images.append(cv2.flip(image,1))
#        measuremants.append(measuremant*-1.0)

#X_train = np.array(images)
#y_train = np.array(measuremants)

#image = augmented_images[1]
#plt.imshow(image)
#cropped = image[60:140, :]
#plt.imshow(cropped)  


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4, verbose=1)


model.save('model_test.h5')

### print the keys contained in the history object
print(history_object.history.keys())
#
#### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
#    
