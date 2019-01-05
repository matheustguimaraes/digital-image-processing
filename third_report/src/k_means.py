import cv2
import numpy as np
from matplotlib import pyplot as plt


def k_means(img, k):
    z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z,
                                    k,
                                    None,
                                    criteria,
                                    10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


if __name__ == '__main__':
    src_img = cv2.imread('../samples/scene_l.bmp')
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    images = [gray_img,
              k_means(gray_img, 2),
              k_means(gray_img, 3),
              k_means(gray_img, 4)]

    fig = plt.figure(1)
    titles = ['Original', 'K = 2', 'K = 3', 'K = 4']
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
    fig.savefig("../results/k_means_figure.png", dpi=300)
