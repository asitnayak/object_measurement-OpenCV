import cv2 as cv
import numpy as np
from object_detector import *

img = cv.imread("Photos/phone.jpg")

cv.imshow('Phone', img)



cv.waitKey(0)