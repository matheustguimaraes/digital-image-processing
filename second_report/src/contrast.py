import cv2
import numpy as np
from matplotlib import pyplot as plt


def contrast_img(img, alpha, beta):
    mul_img = cv2.multiply(img, alpha)
    return np.uint8(cv2.add(mul_img, beta))


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape

    pts1 = np.float32([[1, 0, 100], [0, 1, 20]])
    pts2 = np.float32([[1, 0, 100], [0, 1, 80]])
    pts3 = np.float32([[1, 0, 70], [0, 1, 30]])

    images = [gray_img,
              contrast_img(gray_img, 1.0, 100),
              contrast_img(gray_img, 1.5, 50),
              contrast_img(gray_img, 3.0, 0)]

    fig = plt.figure(1)
    titles = ['Original', 'Contrast 1', 'Contrast 2', 'Contrast 3']
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
    fig.savefig("../results/contrast_figure.jpg", dpi=300)
