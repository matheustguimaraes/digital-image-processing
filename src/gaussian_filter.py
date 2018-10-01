import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import axisartist as ax
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)

def get_gaussian_filter(img, mask_size=3):
    borders = int(mask_size / 2)
    gaussian_image = img[:, :].copy()
    if mask_size == 3:
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])
        for i in range(borders, img.shape[0] - borders):
            for j in range(borders, img.shape[1] - borders):
                kernel = img[i - borders:i + borders + 1, j - borders:j + borders + 1] * gaussian_kernel
                result = np.sum(np.ravel(kernel)) / 16
                gaussian_image[i, j] = result

    elif mask_size == 5:
        gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])
        for i in range(borders, img.shape[0] - borders):
            for j in range(borders, img.shape[1] - borders):
                kernel = img[i - borders:i + borders + 1, j - borders:j + borders + 1] * gaussian_kernel
                result = np.sum(np.ravel(kernel)) / 256
                gaussian_image[i, j] = result
    return gaussian_image


if __name__ == '__main__':
    img = cv2.imread('../samples/baboon.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images = []
    images.append(gray_img)

    gaussian_filter_3x3 = get_gaussian_filter(gray_img, 3)
    images.append(gaussian_filter_3x3)

    gaussian_filter_3x3_2_times = get_gaussian_filter(gaussian_filter_3x3, 3)
    images.append(gaussian_filter_3x3_2_times)

    gaussian_filter_5x5 = get_gaussian_filter(gray_img, 5)
    images.append(gaussian_filter_5x5)

    gaussian_filter_5x5_2_times = get_gaussian_filter(gaussian_filter_5x5, 5)
    images.append(gaussian_filter_5x5_2_times)

    gaussian_filter_5x5_3_times = get_gaussian_filter(gaussian_filter_5x5_2_times, 5)
    images.append(gaussian_filter_5x5_3_times)

    fig = plt.figure(0)
    titles = ['Gray Image',
              '3x3 mask',
              '3x3 mask, 2 times',
              '5x5 mask',
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
