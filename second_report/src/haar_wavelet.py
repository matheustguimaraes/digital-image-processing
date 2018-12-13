import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt


def haar_wavelet(img_data):
    coefficients = pywt.dwt2(img_data, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    return c_a, c_h, c_v, c_d


def haar_waves(img_data, degrees):
    wave = [list(haar_wavelet(img_data))]
    for i in range(degrees - 1):
        wave.append(list(haar_wavelet(wave[-1][0])))
    return wave


if __name__ == '__main__':
    img = cv2.imread('../samples/lena.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    degrees = 10
    subplot = 2

    waves = haar_waves(gray_img, degrees)

    for x in range(len(waves)):
        fig = plt.figure(x)
        for i, img_wave_x in enumerate(waves[x]):
            plt.subplot(subplot,
                        subplot,
                        i + 1), plt.imshow(img_wave_x, 'gray')
    plt.show()
    plt.close()
