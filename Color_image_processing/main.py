import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# In hsv, hue of red = 0, g = 60, b = 120


def color_transform():
    img = cv.imread("./data/Koriand'r.jpg")  # 读入图像
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转换至hsv空间

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('original img')
    plt.show()

    # hair as ROI
    mask = cv.inRange(hsv, np.uint8([150, 50, 50]), np.uint8([179, 255, 255])) + \
           cv.inRange(hsv, np.uint8([0, 50, 85]), np.uint8([2, 255, 255]))  # 选取满足颜色范围的像素为1，生成mask，观察到头发颜色为品红，应在红色和蓝色间
    hair_hsv = cv.bitwise_and(hsv, hsv, mask=mask)  # 取出原图头发部分的信息
    hsv1 = hsv - hair_hsv  # 从原图减去头发
    hair_hsv[:, :, 0] = 60  # 置头发颜色为绿色
    hair_hsv = cv.bitwise_and(hair_hsv, hair_hsv, mask=mask)  # 消去头发外像素的颜色
    hsv1 = hsv1 + hair_hsv  # 将绿色的头发加回原图

    plt.imshow(cv.cvtColor(hsv1, cv.COLOR_HSV2RGB))  # 输出图像
    plt.title('change the color of hair')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # clothes as roi
    # 以下为改变衣服的颜色，原理与上文代码类似
    mask2 = cv.inRange(hsv1, np.uint8([100, 50, 0]), np.uint8([150, 255, 255]))
    clothes_hsv = cv.bitwise_and(hsv1, hsv1, mask=mask2)
    hsv2 = hsv1 - clothes_hsv
    clothes_hsv[:, :, 0] = 0
    clothes_hsv = cv.bitwise_and(clothes_hsv, clothes_hsv, mask=mask2)
    hsv2 = hsv2 + clothes_hsv

    plt.imshow(cv.cvtColor(hsv2, cv.COLOR_HSV2RGB))
    plt.title('chang the color of hair and clothes')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == '__main__':
    color_transform()
