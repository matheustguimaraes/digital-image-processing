import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude


def gradient(img, sigma):
    return gaussian_gradient_magnitude(img, sigma, mode='constant', cval=0.0)


if __name__ == '__main__':
    image = cv2.imread('../samples/lightning.jpg')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    images = [gray_img,
              gradient(gray_img, 2),
              gradient(gray_img, 4),
              gradient(gray_img, 6)]

    fig = plt.figure(1)
    titles = ['Original',
              'Sigma = 1',
              'Sigma = 3',
              'Sigma = 4']
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
    fig.savefig("../results/gradient_figure.jpg", dpi=300)
