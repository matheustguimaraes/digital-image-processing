import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_mean_filter(image, mask_size=3):
    borders = int(mask_size / 2)
    mean_filter = image[:, :].copy()
    for i in range(borders, image.shape[0] - borders):
        for j in range(borders, image.shape[1] - borders):
            kernel = image[i - borders:i + borders + 1, j - borders:j + borders + 1]
            mean_filter[i, j] = int(np.mean(kernel, dtype=np.float32))
    return mean_filter


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images = []
    iterator = 3
    images.append(gray_img)
    for k in range(0, 5):
        images.append(get_mean_filter(gray_img, iterator))
        iterator += 2

    fig = plt.figure(0)
    titles = ['Gray Image', '3x3 mask', '5x5 mask', '7x7 mask', '9x9 mask', '11x11 mask']
    for x in range(len(images)):
        plt.subplot(2, 3, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.axis('off')
        cv2.imwrite("../results/mean_filter_image_mask_{}.jpg".format(x + 1), images[x])

    fig.savefig("../results/mean_filter_figure.jpg")
    plt.show()
    plt.close()
