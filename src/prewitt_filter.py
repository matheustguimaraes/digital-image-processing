import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def set_kernel(direction):
    prewitt_kernel = 0
    if direction:
        prewitt_kernel = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
    elif not direction:
        prewitt_kernel = np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]])
    return prewitt_kernel


def get_prewitt_filter(gray_image):
    borders = int(3 / 2)
    kernel_x = set_kernel(False)
    kernel_y = set_kernel(True)
    prewitt_x = np.zeros(gray_image.shape, dtype=np.int16)
    prewitt_y = np.zeros(gray_image.shape, dtype=np.int16)
    prewitt_image = np.zeros(gray_image.shape, dtype=np.int16)
    for i in range(borders, gray_image.shape[0] - borders):
        for j in range(borders, gray_image.shape[1] - borders):
            prewitt_x[i][j] = np.sum(gray_image[i - borders:i + borders + 1, j - borders:j + borders + 1] * kernel_x)
            prewitt_y[i][j] = np.sum(gray_image[i - borders:i + borders + 1, j - borders:j + borders + 1] * kernel_y)
            prewitt_image[i][j] = math.sqrt(pow(prewitt_y[i][j], 2) + pow(prewitt_x[i][j], 2))
    return prewitt_image, prewitt_y, prewitt_x


if __name__ == '__main__':
    img = cv2.imread('../samples/tiger.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prewitt, prewitt_vertical, prewitt_horizontal = get_prewitt_filter(gray_img)

    images = [gray_img, prewitt, prewitt_vertical, prewitt_horizontal]

    fig = plt.figure(0)
    titles = ['Gray Image', 'Total', 'Vertical', 'Horizontal']
    for x in range(len(images)):
        plt.subplot(2, 2, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/prewitt_filter_image_mask_{}.jpg".format(int(x + 1)), images[x])

    fig.savefig("../results/prewitt_filter_figure.jpg")
    plt.show()
    plt.close()
