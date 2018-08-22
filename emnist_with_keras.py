from collections import Counter

import tensorflow as tf
from tensorflow import data
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
#from sklearn.model_selection import train_test_split
import numpy as np

import dataset
import os

batch_size = 64
num_classes = 62
epochs = 10
img_rows, img_cols = 28, 28

print('Start loading data.')
#Da modificar
folder_path = os.getcwd()
train_dataset = dataset.train(folder_path +'\emnist')
test_dataset = dataset.test(folder_path +'\emnist')
print('Data has been loaded.')

#Non so se le reshape dei tensor x e y contenenti train e test vadano fatte
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices. (Non sono sicuro vadano fatte)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Non so bene cosa faccia, credo non serva
#https://github.com/alexander-fischer/tensorflow-char74-example/blob/master/helpers.py
#train_generator, validation_generator = helpers.create_datagenerator(x_train, x_test, y_train, y_test)

# Convolutional network will be build with Keras.
print('Start training the model.')
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

dataset = dataset.batch(batch_size=batch_size)
dataset = dataset.repeat(epochs)
