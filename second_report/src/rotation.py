import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def rotation(img, degrees, centerR, sca):
    rot = cv2.getRotationMatrix2D(centerR, degrees, sca)
    return cv2.warpAffine(img, rot, img.shape)


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
