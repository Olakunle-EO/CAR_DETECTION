import cv2
import time
import numpy as np

car_cascade = cv2.CascadeClassifier('C:/Users/Fola/Desktop/TOBI/CODES/car detection/haarcascade_car.xml')
cap = cv2.VideoCapture('C:/Users/Fola/Desktop/TOBI/CODES/car detection/4K Video of Highway Traffic!.mp4')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    for (x,y,w,h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('car',plate)

        frames = cv2.resize(frames,(600,400))
        cv2.imshow('Car Detection System', frames)
        # cv2.resizeWindow('Car Detection System', 600, 600)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
cv2.destroyAllWindows()

