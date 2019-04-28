import cv2
import numpy as np
from matplotlib import pyplot as plt


# Restore the correct shape of RMB100 banknote


def preprocess(img):
    """Delete noise area."""
    img[:, 240:261] = 0
    img[:, :20] = 0
    img[:, -15:] = 0
    img[100:125] = 0
    img[235:255] = 0
    img[365:393] = 0
    img[495:520] = 0
    img[625:] = 0
    return img


def test():
    img = cv2.imread('rmb.png', 0)
    threshold = np.vectorize(lambda x: x if x > 30 else 0)
    img = preprocess(img)
    img = threshold(img).astype(np.uint8)
    edges = cv2.Canny(img, 300, 310, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=5, minLineLength=70, maxLineGap=20)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image')

    plt.show()

    line_plot = np.zeros(list(img.shape) + [3], dtype=np.uint8)
    for line in lines:
        line = line.flatten()
        line.astype(np.uint8)
        line = tuple(line)
        cv2.line(line_plot, pt1=line[:2], pt2=line[2:], color=(255, 255, 255), thickness=1)

    plt.imshow(line_plot)
    plt.show()


def corner():
    img = cv2.imread('rmb.png', 0)
    dst = cv2.cornerHarris(img, blockSize=3, ksize=3, k=0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    dst_norm_scaled = (dst_norm_scaled > 45) * 255
    plt.imshow(dst_norm_scaled, cmap='gray')
    plt.show()


def find_corners(idx):  # Note that in OpenCV, point(x, y) = src.at(col, row)
    """
    Find the four corners of a banknote among candidates in idx.
    :param idx: index of all possible corners
    :return: coordinates of four corners of the banknote
    """
    record = {'ul': [], 'ur': [], 'lol': [], 'lor': []}
    idx1 = idx.reshape(-1, 2)
    ul = np.array([0, 0])
    ur = np.array([244, 0])
    lol = np.array([0, 129])
    lor = np.array([244, 129])
    for cor in idx1:
        record['ul'].append(np.linalg.norm(cor - ul))
        record['ur'].append(np.linalg.norm(cor - ur))
        record['lol'].append(np.linalg.norm(cor - lol))
        record['lor'].append(np.linalg.norm(cor - lor))
    # record['ul'].sort()
    # record['ur'].sort()
    # record['lol'].sort()
    # record[]
    ul_cor = np.argmin(record['ul'])
    ur_cor = np.argmin(record['ur'])
    lol_cor = np.argmin(record['lol'])
    lor_cor = np.argmin(record['lor'])
    return np.array([idx1[ul_cor], idx1[ur_cor], idx1[lol_cor], idx1[lor_cor]])


def transform(img, corners_img, chosen_pts=(0, 1, 2)):
    """
    Calculate the corrected image according to the corner image.
    :param img: original image
    :param corners_img: binary image of corner points
    :param chosen_pts: points to be mapped
    :return: the mapped (corrected) image
    """
    # corners_img = cv2.GaussianBlur(corners_img, (5, 5), 0)
    # corners_img = ((corners_img > 0.7 * corners_img.max()) * 255).astype(np.uint8)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.imshow(corners_img, cmap='gray')
    # plt.show()
    idx = cv2.findNonZero(corners_img)
    pts1 = find_corners(idx)
    pts2 = np.array([[0, 0], [200, 0], [0, 100], [200, 100]])  # RMB100 = (155, 77), ratio = 2.01299
    M = cv2.getAffineTransform(pts1[chosen_pts,].astype(np.float32), pts2[chosen_pts,].astype(np.float32))
    out = cv2.warpAffine(img, M, (200, 100))
    plt.imshow(out, cmap='gray')
    plt.show()
    return out


def main():
    # preprocess the image
    img = cv2.imread('rmb.png', 0)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = preprocess(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    imgs = dict()
    corners_plots = dict()
    # imgs[(0, 0)] = img[:120, 15:250]
    # plt.imshow(imgs[(0, 0)], cmap='gray')
    # plt.show()
    int_point = [0, 120, 240, 380, 510, 640]
    i = 1
    for row in range(5):
        for col in range(2):
            imgs[(row, col)] = img[int_point[row]:int_point[row + 1], 250 * col:250 * (col + 1)]
            plt.subplot(5, 2, i), plt.imshow(imgs[(row, col)], cmap='gray')
            i += 1
    plt.show()

    i = 1
    for row in range(5):
        for col in range(2):
            corners_plot = cv2.cornerHarris(imgs[(row, col)], blockSize=5, ksize=3, k=0.04)
            dst_norm = np.empty(corners_plot.shape, dtype=np.float32)
            cv2.normalize(corners_plot, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dst_norm_scaled = cv2.convertScaleAbs(dst_norm)  # convert to uint8
            _, dst_norm_scaled = cv2.threshold(dst_norm_scaled, 100, 255, type=cv2.THRESH_BINARY)  #
            corners_plots[(row, col)] = dst_norm_scaled.astype(np.uint8)
            plt.subplot(5, 2, i), plt.imshow(dst_norm_scaled, cmap='gray')
            i += 1
    plt.show()

    # inspect all dst_norm_scaled
    # for row in range(5):
    #     for col in range(2):
    #         plt.imshow(corners_plots[(row, col)], cmap='gray'), plt.show()

    exception = {(0, 0): [0, 2, 3], (0, 1): [1, 2, 3], (4, 0): [0, 1, 3]}  # Manually find the corners used to calculate
    # the transform matrix since some of the four has not been detected.
    outs = exception.copy()  # Output images.
    for row in range(5):
        for col in range(2):
            if (row, col) in exception:
                img1, corner_plot1 = imgs[(row, col)], corners_plots[(row, col)]
                out = transform(img1, corner_plot1, chosen_pts=exception[row, col])
            else:
                img1, corner_plot1 = imgs[(row, col)], corners_plots[(row, col)]
                out = transform(img1, corner_plot1)
            outs[(row, col)] = out

    # Recover the RGB images using their 3 components
    rgb1 = np.empty([100, 200, 3], dtype=np.uint8)
    rgb2 = np.empty([100, 200, 3], dtype=np.uint8)

    rgb1[:, :, 0] = outs[(0, 0)]
    rgb1[:, :, 1] = outs[(1, 0)]
    rgb1[:, :, 2] = outs[(2, 0)]

    rgb2[:, :, 0] = outs[(0, 1)]
    rgb2[:, :, 1] = outs[(1, 1)]
    rgb2[:, :, 2] = outs[(2, 1)]

    plt.subplot(211), plt.imshow(rgb1)
    plt.subplot(212), plt.imshow(rgb2)
    plt.show()
    pass


if __name__ == '__main__':
    main()
