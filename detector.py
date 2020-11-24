import cv2
import os
import numpy as np
from PIL import Image
import pickle
import sqlite3

recognizer=cv2.face.LBPHFaceRecognizer_create()
#rec=cv2.createLBPHFaceRecognizer();
recognizer.read('trainner/trainingData.yml')
cascadePath="E:/My_Dev/haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadePath);
path='dataSet'

def getProfile(id):
    conn=sqlite3.connect('faceBase.db')
    cmd="SELECT * FROM people WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
    
#id=0
cam=cv2.VideoCapture(0)
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
font=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(255,255,0)
while True:
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #faces=faceCascade.detectMultiScale(gray,1.3,5);
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5); #,minSize=(100, 100), flags=cv2.CascadeClassifier
    for(x,y,w,h) in faces:
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        profile=getProfile(id)
        if (profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),font,fontscale,fontcolor)                    
        cv2.imshow("face",img)
        cv2.waitKey(10)
