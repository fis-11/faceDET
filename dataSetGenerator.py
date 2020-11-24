import cv2
import numpy as np
import sqlite3

cam=cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('E:/My_Dev/haarcascade_frontalface_default.xml');

def insertUpdate(Id, Name):
    conn=sqlite3.connect("E:/My_Dev/faceBase.db")
    cmd="SELECT * FROM People WHERE Id="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
        if(isRecordExist==1):
            cmd="UPDATE People SET Name="+str(Name)+" WHERE Id="+str(Id)
        else:
            cmd="INSERT INTO People(Id, Name) Values("+str(Id)+","+str(Name)+")"
        conn.execute(cmd)
        conn.commit()
        #conn.close()
    
id=input('enter user id : ')
name=input('enter user name : ')
insertUpdate(id, name)
sampleNum=0;
while True:
    ret, img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/user."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("face",img);
        cv2.waitKey(100)
        if(sampleNum>20):
            cam.release()
            cv2.destroyAllWindows()
            break
