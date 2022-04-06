import cv2 as cv
import numpy as np
from object_detector import *

# Load Aruco detector
parameters = cv.aruco.DetectorParameters_create()
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)


# Load object detector
detector = HomogeneousBgDetector()

# Load Cap
cap = cv.VideoCapture(0)


while True:
    _, img = cap.read()

    # Get aruco marker
    corners, _, _ = cv.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    print(corners)

    if corners:

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv.polylines(img, int_corners, True, (0,255,0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv.arcLength(corners[0], True)

        # Pixel to CM ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(img)

        # Draw objects boundries
        for cnt in contours:
            # Draw polygon
            #cv.polylines(img, [cnt], True, (255,0,0), 2)

            # Get rect
            rect = cv.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get width and height of the objects by applying the ratio, pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio
            
            # Display rectangle
            box = cv.boxPoints(rect)
            box = np.int0(box)

            cv.circle(img, (int(x), int(y)), 5, (0,0,255), -1)
            cv.polylines(img, [box], True, (255,0,0), 2)
            cv.putText(img, f"width {round(object_width, 1)} cm", (int(x-100), int(y - 20)), cv.FONT_HERSHEY_PLAIN, 2, (100,200,0), 2)
            cv.putText(img, f"Length {round(object_height, 1)} cm", (int(x-100), int(y + 15)), cv.FONT_HERSHEY_PLAIN, 2, (100,200,0), 2)


    cv.imshow('Image', img)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()