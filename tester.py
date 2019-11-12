import cv2
import os
import numpy as ny
import faceRecognition as fr

test_img=cv2.imread('D:/FaceRecognition/TestImages/pai2.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detested:",faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

resized_img=cv2.resize(test_img,(960,636))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows