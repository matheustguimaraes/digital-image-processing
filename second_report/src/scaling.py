import cv2
import numpy as np
from matplotlib import pyplot as plt


def scaling(gray, fx, fy):
    return cv2.resize(gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rows, img_cols = gray_img.shape

    fx1 = 3
    fx2 = 2

    fy1 = 2
    fy2 = 4
    fy3 = 1

    images = [gray_img,
              scaling(gray_img, fx1, fy1),
              scaling(gray_img, fx2, fy2),
              scaling(gray_img, fx1, fy3)]

    fig = plt.figure(1)
    titles = ['Original', 'Scaling 1', 'Scaling 2', 'Scaling 3']
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
    fig.savefig("../results/scaling/scaling_figure.jpg", dpi=300)
