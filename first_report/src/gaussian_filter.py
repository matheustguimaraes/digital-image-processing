import cv2
import numpy as np
from matplotlib import pyplot as plt


def set_mask_size(mask_size):
    gaussian_kernel = 0
    division = 0
    if mask_size == 3:
        division = 16
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])
    elif mask_size == 5:
        division = 256
        gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])
    return gaussian_kernel, division


def get_gaussian_filter(gray, mask_size=3):
    bd = int(mask_size / 2)
    gaussian_image = gray[:, :].copy()
    gaussian_k, division = set_mask_size(mask_size)
    for i in range(bd, gray.shape[0] - bd):
        for j in range(bd, gray.shape[1] - bd):
            kernel = gray[i - bd:i + bd + 1, j - bd:j + bd + 1] * gaussian_k
            result = np.sum(np.ravel(kernel)) / division
            gaussian_image[i, j] = result
    return gaussian_image


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filter_3x3 = get_gaussian_filter(gray_img, 3)
    filter_3x3_2_times = get_gaussian_filter(filter_3x3, 3)
    filter_5x5 = get_gaussian_filter(gray_img, 5)
    filter_5x5_2_times = get_gaussian_filter(filter_5x5, 5)
    filter_5x5_3_times = get_gaussian_filter(filter_5x5_2_times, 5)

    images = [gray_img,
              filter_3x3,
              filter_3x3_2_times,
              filter_5x5,
              filter_5x5_2_times,
              filter_5x5_3_times]

    fig = plt.figure(0)
    titles = ['Gray Image',
              '3x3 mask, 1 time',
              '3x3 mask, 2 times',
              '5x5 mask, 1 time',
              '5x5 mask, 2 times',
              '5x5 mask, 3 times']

    for x in range(len(images)):
        plt.subplot(2, 3, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/gaussian/{}.jpg".format(titles[x]), images[x])

    fig.savefig("../results/gaussian/figure.jpg")
    plt.show()
    plt.close()

