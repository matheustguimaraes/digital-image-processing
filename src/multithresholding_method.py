import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_multithresholding_image(gray_image, t_min, t_max):
    thresholding_image = np.zeros(gray_image.shape)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            thresholding_image[i][j] = 255 if ((gray_image[i][j] >= t_min) & (gray_image[i][j] < t_max)) else 0
    return thresholding_image


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold_80_to_120 = get_multithresholding_image(gray_img, 80, 120)
    threshold_150_to_200 = get_multithresholding_image(gray_img, 150, 200)
    threshold_200_to_255 = get_multithresholding_image(gray_img, 200, 255)

    images = [gray_img, threshold_80_to_120, threshold_150_to_200, threshold_200_to_255]

    fig = plt.figure(0)
    titles = ['Gray Image', 'Threshold = 80 to 120', 'Threshold = 150 to 200', 'Threshold = 200 to 255']
    for x in range(len(images)):
        plt.subplot(2, 2, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/multithresholding_image_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/multithresholding_method_figure.jpg")
    plt.show()
    plt.close()
