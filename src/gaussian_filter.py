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


def get_gaussian_filter(gray_image, mask_size=3):
    borders = int(mask_size / 2)
    gaussian_image = gray_image[:, :].copy()
    gaussian_kernel, division = set_mask_size(mask_size)
    for i in range(borders, gray_image.shape[0] - borders):
        for j in range(borders, gray_image.shape[1] - borders):
            kernel = gray_image[i - borders:i + borders + 1, j - borders:j + borders + 1] * gaussian_kernel
            result = np.sum(np.ravel(kernel)) / division
            gaussian_image[i, j] = result
    return gaussian_image


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian_filter_3x3 = get_gaussian_filter(gray_img, 3)
    gaussian_filter_3x3_2_times = get_gaussian_filter(gaussian_filter_3x3, 3)
    gaussian_filter_5x5 = get_gaussian_filter(gray_img, 5)
    gaussian_filter_5x5_2_times = get_gaussian_filter(gaussian_filter_5x5, 5)
    gaussian_filter_5x5_3_times = get_gaussian_filter(gaussian_filter_5x5_2_times, 5)

    images = [gray_img,
              gaussian_filter_3x3,
              gaussian_filter_3x3_2_times,
              gaussian_filter_5x5,
              gaussian_filter_5x5_2_times,
              gaussian_filter_5x5_3_times]

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
        cv2.imwrite("../results/gaussian_filter_image_mask_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/gaussian_filter_figure.jpg")
    plt.show()
    plt.close()
