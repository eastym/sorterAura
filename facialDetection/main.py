import cv2
import numpy


face_cascade_path = '/usr/local/Cellar/opencv/4.7.0_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)

src = cv2.imread('/Users/eastym/aura/FNtpMRdagAIVJrS.jpeg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(src_gray)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
cv2.imwrite('/Users/eastym/auraSorter/FNtpMRdagAIVJrS.png', src)