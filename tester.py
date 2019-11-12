import cv2
import os
import numpy as ny
import faceRecognition as fr

test_img=cv2.imread('D:/FaceRecognition/TestImages/testimage.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detested:",faces_detected)

faces,faceID=fr.label_for_training_data('D:/FaceRecognition/trainingimages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')
name={0:'Pai',1:'Nice'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print('confidence: ',confidence)
    print('label: ',label)
    fr.darw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows