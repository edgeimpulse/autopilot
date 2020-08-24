import sys, os, shutil, signal, random, operator, functools, time, subprocess, math
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Lambda, Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from PIL import Image
from PIL import ImageEnhance
import tensorflow.keras.backend as K
import itertools
from scipy.stats import norm

def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

train_dataset = train_dataset.map(augment_image, tf.data.experimental.AUTOTUNE)

def reshape(image, label):
    return tf.reshape(image, (120, 160, 1)), label
train_dataset = train_dataset.map(reshape, tf.data.experimental.AUTOTUNE)

BATCH_SIZE = 128
train_dataset, validation_dataset = set_batch_size(BATCH_SIZE, train_dataset, validation_dataset)

mat = np.empty((7,7))
r = [-1000, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1000]
for (i,j) in itertools.product(range(7), range(7)):
    mat[i][j] = norm.cdf(r[j + 1], loc=i) - norm.cdf(r[j], loc=i)
mat = K.constant(mat)

def loss(y_true, y_pred):
    diff = (K.dot(y_true, mat) - y_pred)
    return K.sum(K.sum(diff * diff, axis=1))

c = K.constant(np.array([-3, -2, -1, 0, 1, 2, 3]))
def acc(y_true, y_pred):
    diff = (K.sum(y_true * c, axis=1) - K.sum(y_pred * c, axis=1))
    return (K.mean(diff * diff))

model = Sequential()
model.add(InputLayer(input_shape=(120, 160, 1), name='x_input'))
model.add(Conv2D(16, 3, 2, activation='relu', padding='same'))
model.add(Conv2D(16, 3, 2, activation='relu', padding='same'))
model.add(Conv2D(32, 3, 2, activation='relu', padding='same'))
model.add(Conv2D(32, 3, 1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer='l1'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu', kernel_regularizer='l1'))
model.add(Dense(7, activation='softmax'))

model.compile(loss=loss,
              optimizer=Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.999),
              metrics=[acc])

early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10,
                                     verbose=0, mode='min', restore_best_weights=True)

history = model.fit(train_dataset, epochs=200, validation_data=validation_dataset,
             verbose=2, use_multiprocessing=True, workers=4,
             callbacks=[early_stop])