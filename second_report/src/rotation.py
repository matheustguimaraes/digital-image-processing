import cv2
import numpy as np
from matplotlib import pyplot as plt


def rotation(gray, pt1, pt2, cols, rows):
    matrix = cv2.getAffineTransform(pt1, pt2)
    return cv2.warpAffine(gray, matrix, (cols, rows))


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    pts3 = np.float32([[50, 50], [150, 50], [120, 200]])
    pts4 = np.float32([[10, 100], [80, 50], [180, 250]])

    images = [gray_img,
              rotation(gray_img, pts1, pts2, img_cols, img_rows),
              rotation(gray_img, pts2, pts3, img_cols, img_rows),
              rotation(gray_img, pts2, pts4, img_cols, img_rows)
              ]

    fig = plt.figure(1)
    titles = ['Original', 'Rotation 1', 'Rotation 2', 'Rotation 3']
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
    fig.savefig("../results/rotation_figure.jpg", dpi=300)
