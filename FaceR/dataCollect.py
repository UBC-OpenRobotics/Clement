import cv2
import os
import time

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input('\n enter username and press <return>')
path = "dataset/"+name
if not os.path.isdir(path):
    os.mkdir(path)

count = 0
timeTab = []
while(True):
    start_time = time.time()
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(face) > 0:
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]
        cv2.imwrite(path+"/"+str(count)+".jpg",gray[y:y+h,x:x+w])
    count+=1
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200: # Take 30 face sample and stop video
         break

cam.release()
cv2.destroyAllWindows()
