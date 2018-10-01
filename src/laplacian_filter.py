import cv2
import numpy as np
from matplotlib import pyplot as plt


def set_kernel(type=3):
    laplacian_kernel = 0
    if type == 3:
        laplacian_kernel = np.array([[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]])
    elif type == 5:
        laplacian_kernel = np.array([[-1, -1, -1, -1, -1],
                                     [-1, -1, -1, -1, -1],
                                     [-1, -1, 24, -1, -1],
                                     [-1, -1, -1, -1, -1],
                                     [-1, -1, -1, -1, -1]])
    return laplacian_kernel


def get_laplacian_filter(image, type=3):
    borders = int(type / 2)
    kernel = set_kernel(type)
    laplacian_image = image[:, :].copy()
    for i in range(borders, image.shape[0] - borders):
        for j in range(borders, image.shape[1] - borders):
            laplacian_result = np.multiply(image[i - borders:i + borders + 1, j - borders:j + borders + 1], kernel)
            result = np.sum(laplacian_result)
            laplacian_image[i, j] = result
    return laplacian_image
    # return cv2.normalize(laplacian_image, 0, 255, cv2.NORM_MINMAX)


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images = []
    images.append(gray_img)

    laplacian_filter_3x3 = get_laplacian_filter(gray_img, 3)
    images.append(laplacian_filter_3x3)

    # laplacian_filter_3x3_2_times = get_laplacian_filter(laplacian_filter_3x3, 3)
    # images.append(laplacian_filter_3x3_2_times)

    laplacian_filter_5x5 = get_laplacian_filter(gray_img, 5)
    images.append(laplacian_filter_5x5)

    # laplacian_filter_5x5_2_times = get_laplacian_filter(laplacian_filter_5x5, 5)
    # images.append(laplacian_filter_5x5_2_times)
    #
    # laplacian_filter_5x5_3_times = get_laplacian_filter(laplacian_filter_5x5_2_times, 5)
    # images.append(laplacian_filter_5x5_3_times)

    fig = plt.figure(0)
    titles = ['Gray Image',
              '3x3 mask',
              # '3x3 mask, 2 times',
              '5x5 mask',
              # '5x5 mask',
              # '5x5 mask, 2 times',
              # '5x5 mask, 3 times'
              ]
    for x in range(len(images)):
        plt.subplot(1, 3, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/laplacian_filter_image_mask_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/laplacian_filter_figure.jpg")
    plt.show()
    plt.close()
