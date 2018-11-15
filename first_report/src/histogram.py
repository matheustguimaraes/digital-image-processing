import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_histogram(gray_image):
    hist = np.zeros(256)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            hist[gray_image[i, j]] += 1
    return hist


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    histogram = get_histogram(gray_img)

    images = [gray_img, histogram]

    fig = plt.figure(0)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.axis('off')
    ax_u = ax1.imshow(gray_img, cmap='gray')
    plt.title('Gray Image')

    ax2 = fig.add_subplot(1, 2, 2)
    x_values = np.arange(histogram.shape[0])
    bar = ax2.bar(x_values, histogram, color='k')
    divider = make_axes_locatable(ax2)
    plt.xlim([0, 256])
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title('Histogram')

    fig.savefig("../results/histogram/histogram_figure.jpg", dpi=300)
    plt.show()
    plt.close()
