#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

PATH = "features/"

people = [person for person in os.listdir(PATH)]
test = []
test_labels = []
train = []
train_labels = []

names = {}

split = 0.3

count = -1
for feat in people:
    count += 1
    names[count] = str(feat)
    fileName = PATH + str(feat)
    file_in = open(fileName,"rb")
    fullArr = pickle.load(file_in)
    nbFeat = -1
    for arr in fullArr:
        nbFeat += 1
        if nbFeat/len(fullArr) < 1 - split:
            train.append(arr)
            train_labels.append(count)
        else:
            test.append(arr)
            test_labels.append(count)

names_file = open("names","wb")
pickle.dump(names,names_file)
names_file.close()

train = np.asarray(train)
test = np.asarray(test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,2048)),
    keras.layers.Dense(7,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=10)

model.save('pouet.model')
