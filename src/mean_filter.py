import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_mean_filter(img, mask_size=3):
    borders = int(mask_size / 2)
    mean_filter = img[:, :].copy()
    for i in range(borders, img.shape[0] - borders):
        for j in range(borders, img.shape[1] - borders):
            kernel = img[i - borders:i + borders + 1, j - borders:j + borders + 1]
            mean_filter[i, j] = int(np.mean(kernel, dtype=np.float32))
    return mean_filter


if __name__ == '__main__':
    img = cv2.imread('../samples/baboon.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images = []
    images.append(gray_img)
    images.append(get_mean_filter(gray_img, 3))
    images.append(get_mean_filter(gray_img, 5))
    images.append(get_mean_filter(gray_img, 7))
    images.append(get_mean_filter(gray_img, 9))
    images.append(get_mean_filter(gray_img, 11))

    fig = plt.figure(0)
    fig.canvas.set_window_title('Mean Filter')
    titles = ['Src Image', '3x3 Mask', '5x5 Mask', '7x7 Mask', '9x9 Mask', '11x11 Mask']
    for x in range(len(images)):
        plt.subplot(2, 3, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        cv2.imwrite("../results/mean_filter_image_mask_{}.jpg".format(x + 1), images[x])

    plt.show()
    plt.close()
