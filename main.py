import cv2 as cv
import numpy as np

webcam = True
path = "sheet.jpg"
cap = cv.VideoCapture(1)
cap.set(10, 160)
cap.set(3, 400)
cap.set(4, 400)
scale=3
wP = 210*scale
hP = 297*scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv.imread(path)

    #imgContours, counts = 
