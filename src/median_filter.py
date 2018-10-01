import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_median_filter(img, mask_size=3):
    borders = int(mask_size / 2)
    median_filter = img[:, :].copy()
    for i in range(borders, img.shape[0] - borders):
        for j in range(borders, img.shape[1] - borders):
            kernel = np.ravel(img[i - borders:i + borders + 1, j - borders:j + borders + 1])
            median = sorted(kernel)[int(((mask_size * mask_size) / 2) + 1)]
            median_filter[i, j] = median
    return median_filter


if __name__ == '__main__':
    image = cv2.imread('../samples/baboon.jpg')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    images = []
    iterator = 3
    images.append(gray_img)
    for k in range(0, 5):
        images.append(get_median_filter(gray_img, iterator))
        iterator += 2

    fig = plt.figure(0)
    titles = ['Gray image', '3x3 mask', '5x5 mask', '7x7 mask', '9x9 mask', '11x11 mask']
    for x in range(len(images)):
        plt.subplot(2, 3, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/median_filter_image_mask_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/median_filter_figure.jpg")
    plt.show()
    plt.close()
