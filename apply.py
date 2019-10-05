import tensorflow as tf
import cv2
import os

IMG_SIZE = 150
img_array = cv2.imread('Dataset/validation/clement/User.1.60.jpg',cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

new_array = new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("pouet.model")

prediction = model.predict(new_array)

print(prediction)
