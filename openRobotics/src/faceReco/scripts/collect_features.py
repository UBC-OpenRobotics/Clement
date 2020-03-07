#!/usr/bin/env python
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import numpy as np
import os
import pickle


features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')

PATH = "dataset/"
PATH2 = "features/"

people = [person for person in os.listdir(PATH)]

for person in people:
    fileName = PATH2 + str(person)

    if not os.path.isfile(fileName):
        im_features = []

        for im in os.listdir(PATH+str(person)):
            img = image.load_img(PATH+str(person)+"/"+str(im),target_size=(224,224))
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis=0)
            x = utils.preprocess_input(x,version=1)
            im_features.append(features.predict(x))

        file_out = open(fileName,"wb")
        pickle.dump(im_features,file_out)
        file_out.close()
