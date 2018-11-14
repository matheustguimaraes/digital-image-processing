import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_thresholding_image(gray_image, threshold):
    thresholding_image = np.zeros(gray_image.shape)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            thresholding_image[i][j] = 255 if (gray_img[i][j] >= threshold) else 0
    return thresholding_image


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold_80 = get_thresholding_image(gray_img, 80)
    threshold_150 = get_thresholding_image(gray_img, 150)
    threshold_200 = get_thresholding_image(gray_img, 200)

    images = [gray_img, threshold_80, threshold_150, threshold_200]

    fig = plt.figure(0)
    titles = ['Gray Image', 'Threshold = 80', 'Threshold = 150', 'Threshold = 200']
    for x in range(len(images)):
        plt.subplot(2, 2, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/thresholding_image_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/thresholding_method_figure.jpg")
    plt.show()
    plt.close()
