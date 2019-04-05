import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# In hsv, hue of red = 0, g = 60, b = 120


def color_transform():
    img = cv.imread("./data/starfire.jpg")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # hair as ROI
    mask = cv.inRange(hsv, np.uint8([150, 50, 50]), np.uint8([179, 255, 255])) + \
           cv.inRange(hsv, np.uint8([0, 50, 85]), np.uint8([2, 255, 255]))
    hair_hsv = cv.bitwise_and(hsv, hsv, mask=mask)
    hsv = hsv - hair_hsv
    hair_hsv[:, :, 0] = 60
    hair_hsv = cv.bitwise_and(hair_hsv, hair_hsv, mask=mask)
    hsv = hsv + hair_hsv
    plt.imshow(cv.cvtColor(hsv, cv.COLOR_HSV2RGB))
    plt.show()


if __name__ == '__main__':
    color_transform()
