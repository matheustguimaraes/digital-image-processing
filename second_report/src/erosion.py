import cv2
import numpy as np
from matplotlib import pyplot as plt


def erosion_img(gray, kernel):
    return cv2.erode(gray, kernel, iterations=2)


if __name__ == '__main__':
    img = cv2.imread('../samples/objects.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape
    gray_img = 255 - gray_img

    kernel3 = np.array([1, 1, 1] * 3)
    kernel5 = np.array([1, 1, 1] * 5)
    kernel7 = np.array([1, 1, 1] * 7)

    images = [gray_img,
              erosion_img(gray_img, kernel3),
              erosion_img(gray_img, kernel5),
              erosion_img(gray_img, kernel7)]

    fig = plt.figure(1)
    titles = ['Original',
              'Erosion 1',
              'Erosion 2',
              'Erosion 3']

    for i in range(0, len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
        plt.subplots_adjust(left=0.0,
                            bottom=0.05,
                            right=1.0,
                            top=0.95,
                            wspace=0.0)

    plt.show()
    fig.savefig("../results/erosion_figure.jpg", dpi=300)
