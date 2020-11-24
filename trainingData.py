import os
import cv2
import numpy as np
from PIL import Image

#recognizer=cv2.createLBPHFaceRecognizer();
recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def getImageID(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imageP in imagePath:
        faceImg=Image.open(imageP).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imageP)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)

        cv2.imshow("Training...",faceNp)
        cv2.waitKey(10)

    return np.array(IDs), faces

IDs, faces=getImageID(path)
recognizer.train(faces,IDs)
recognizer.save('trainner/trainingData.yml')
cv2.destroyAllWindows()
