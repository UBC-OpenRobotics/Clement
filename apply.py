import tensorflow as tf
import cv2
import os
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("pouet.model")
font = cv2.FONT_HERSHEY_SIMPLEX

names = ["Celine","Sam","Harry","Marika","Clement"]

IMG_SIZE = 150

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        new_array = cv2.resize(gray[y:y+h,x:x+w],(IMG_SIZE,IMG_SIZE))
        new_array = new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
        prediction = model.predict(new_array)
        cv2.putText(img, names[np.argmax(prediction)], (x+5,y-5), font, 1, (255,255,255), 2)
        
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
