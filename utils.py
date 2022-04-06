import cv2 as cv
import numpy as np

def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv.cvtColor(img, cv.COLOR_BayerRG2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv.dialate(imgCanny, kernel, iterations=3)
    imgThre = cv.erode(imgDial, kernel, iterations=2)

    if showCanny:
        cv.imshow('Canny', imgThre)

    contours, hiearchy = cv.findContours(imgThre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv.contourArea(i)
        if area > minArea:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02*peri, True)
            bbox = cv.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key = lambda x:x[1], reverse = True)
    if draw:
        for con in finalContours:
            cv.drawContours(img, con[4], -1, (0,0,255), 3)
    return img, finalContours