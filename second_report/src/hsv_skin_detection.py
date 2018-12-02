import cv2
import numpy as np
from matplotlib import pyplot as plt


def skin_detection(gray):
    lower = np.array([50,50,50], dtype="uint8")
    upper = np.array([180,180,220], dtype="uint8")

    skin_mask = cv2.inRange(gray, lower, upper)

    return cv2.bitwise_and(src_img, src_img, mask=skin_mask)


if __name__ == '__main__':
    src_img = cv2.imread('../samples/matheus.jpg')
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)

    fig = plt.figure(1)
    plt.subplot(1, 2, 1), plt.imshow(gray_img, 'gray')
    plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(skin_detection(gray_img), 'hot')
    plt.axis('off')
    plt.show()

    fig.savefig("../results/hsv_skin_figure.jpg", dpi=300)

