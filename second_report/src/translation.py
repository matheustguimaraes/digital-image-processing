import cv2
import numpy as np
from matplotlib import pyplot as plt


def translation(gray, matrix, cols, rows):
    return cv2.warpAffine(gray, matrix, (cols, rows))


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape

    pts1 = np.float32([[1, 0, 100], [0, 1, 20]])
    pts2 = np.float32([[1, 0, 100], [0, 1, 80]])
    pts3 = np.float32([[1, 0, 70], [0, 1, 30]])

    images = [gray_img,
              translation(gray_img, pts1, img_cols, img_rows),
              translation(gray_img, pts2, img_cols, img_rows),
              translation(gray_img, pts3, img_cols, img_rows)
              ]

    fig = plt.figure(1)
    titles = ['Original', 'Translation 1', 'Translation 2', 'Translation 3']
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
    fig.savefig("../results/translation_figure.jpg", dpi=300)
