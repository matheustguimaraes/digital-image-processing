import cv2
import numpy as np
from matplotlib import pyplot as plt


def gamma_correction(gray, correction):
    gray = gray / 255.0
    gray = cv2.pow(gray, correction)
    return np.uint8(gray * 255)


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape

    images = [gray_img,
              gamma_correction(gray_img, 0.1),
              gamma_correction(gray_img, 1.5),
              gamma_correction(gray_img, 3.0)]

    fig = plt.figure(1)
    titles = ['Original',
              'Gamma correction 1',
              'Gamma correction 2',
              'Gamma correction 3']

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
    fig.savefig("../results/brightness_figure.jpg", dpi=300)
