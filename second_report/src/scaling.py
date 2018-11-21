import cv2
import numpy as np
from matplotlib import pyplot as plt


def scaling(gray, M, cols, rows):
    return cv2.warpAffine(gray, M, (cols, rows))


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    M = np.float32([[1, 0, 100], [0, 1, 50]])
    M2 = np.float32([[1, 0, 10], [0, 1, 40]])
    M3 = np.float32([[1, 0, 50], [0, 1, 10]])
    rows, cols = gray_img.shape

    images = [gray_img,
              scaling(gray_img, M, cols, rows),
              scaling(gray_img, M2, cols, rows),
              scaling(gray_img, M3, cols, rows)]

    fig = plt.figure(1)
    titles = ['Original', 'Scaling 1', 'Scaling 2', 'Scaling 3']
    for i in range(0, len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95,
                            wspace=0.0)

    plt.show()
    fig.savefig("../results/scaling/scaling_figure.jpg", dpi=300)
