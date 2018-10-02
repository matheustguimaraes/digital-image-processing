import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_image_equalization(gray_image):
    min_gray_value = gray_image.min()
    max_gray_value = gray_image.max()
    difference = int(max_gray_value - min_gray_value)
    equalized_image = np.zeros(gray_image.shape)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            equalized_image[i, j] = ((gray_image[i, j] - min_gray_value) * 255) / difference
    return equalized_image


def get_histogram(gray_image):
    hist = np.zeros(256)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            hist[int(gray_image[i, j])] += 1
    return hist


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    histogram = get_histogram(gray_img)
    eq_image = get_image_equalization(gray_img)
    histogram_equalization = get_histogram(eq_image)

    images = [gray_img, histogram, eq_image, histogram]

    fig = plt.figure(0)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.axis('off')
    ax1.imshow(gray_img, cmap='gray')
    plt.title('Gray Image')

    ax2 = fig.add_subplot(2, 2, 2)
    x_values = np.arange(histogram.shape[0])
    ax2.bar(x_values, histogram, color='k')
    make_axes_locatable(ax2)
    plt.xlim([0, 256])
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    ax3 = fig.add_subplot(2, 2, 3)
    plt.axis('off')
    ax3.imshow(eq_image, cmap='gray')
    plt.title('Equalized Image')

    ax4 = fig.add_subplot(2, 2, 4)
    x_values2 = np.arange(histogram_equalization.shape[0])
    ax4.bar(x_values2, histogram_equalization, color='k')
    make_axes_locatable(ax4)
    plt.xlim([0, 256])
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    fig.savefig("../results/equalized_histogram_figure.jpg", dpi=300)
    plt.show()
    plt.close()
