from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
modelLin = tf.keras.models.load_model("pouet.model")

name_file = open("names","rb")
names = pickle.load(name_file)
name_file.close()
print(names)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
font = cv2.FONT_HERSHEY_SIMPLEX

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if os.path.isfile("features_file.pickle"):
    prev_feat = open("features_file.pickle","rb")
    feats = pickle.load(prev_feat)
    prev_feat.close()

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        arr = cv2.resize(img[y:y+h,x:x+w],(224,224))
        arr = image.img_to_array(arr)
        arr = np.expand_dims(arr,axis=0)
        arr = utils.preprocess_input(arr,version=1)
        prediction = model.predict(arr)
        prediction = prediction.reshape(1,1,2048)
        finalPred = modelLin.predict(prediction)

        cv2.putText(img, names[np.argmax(finalPred)], (x+5,y-5), font, 1, (255,255,255), 2)


    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
