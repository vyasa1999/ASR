# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:38:16 2021

@author: 1999a
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from dataload import loadPreprocess
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
features, labels = loadPreprocess()
enc = OneHotEncoder()
labels = enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = 42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state = 16)
x_train = tf.expand_dims(x_train, axis=-1)
x_valid = tf.expand_dims(x_valid, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 172, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(31, activation="softmax"))
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_valid, y_valid))

test = np.argmax(model.predict(x_test), axis=-1)

print(metrics.accuracy_score(np.argmax(y_test, axis=-1), test))