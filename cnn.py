from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# path_train = 'Dataset/train'
# path_val = 'Dataset/validation'
#
# train_beau_dir = os.path.join(train_dir, 'beau')  # directory with our training cat pictures
# train_clement_dir = os.path.join(train_dir, 'clement')  # directory with our training dog pictures
# validation_beau_dir = os.path.join(validation_dir, 'beau')  # directory with our validation cat pictures
# validation_clement_dir = os.path.join(validation_dir, 'clement')  # directory with our validation dog pictures
#
# num_beau_tr = len(os.listdir(train_beau_dir))
# num_clement_tr = len(os.listdir(train_clement_dir))
#
# num_beau_val = len(os.listdir(validation_beau_dir))
# num_clement_val = len(os.listdir(validation_clement_dir))
#
# total_train = num_beau_tr + num_clement_tr
# total_val = num_beau_val + num_clement_val

PATH = "Dataset/"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

batch_size = 1
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse',
                                                           color_mode = 'grayscale')

# total_train = 0
# path_train = 'Dataset/train'
# for dir in os.listdir(path_train):
#     total_train += len([name for name in os.listdir(os.path.join(path_train,dir))])

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='sparse',
                                                              color_mode = 'grayscale')

# total_val = 0
# path_validation = 'Dataset/validation'
# for dir in os.listdir(path_validation):
#     total_val += len([name for name in os.listdir(os.path.join(path_validation,dir))])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    #steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    #validation_steps=total_val // batch_size
)

model.save('pouet.model')

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
