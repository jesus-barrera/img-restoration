# -*- coding: utf-8 -*-

import numpy as np
import random
import sys

from PIL import Image

from mlp import MultiLayerPerceptron as MLP

NOISE_LEVEL = 0.20
IMAGE_PATH = 'lenna.jpg'

def add_noise(img, points):
    pixels = img.load()

    for x, y in points:
        pixels[x, y] = 255 # white points

    return img

def generate_training_set(img, points):
    pixels = img.load()
    training_set = []

    for x, y in points:
        gray = pixels[x, y]

        inputs = np.array([x, y])
        output = int2bin(gray)
        training_set.append((inputs, output))

    return training_set

def int2bin(value):
    binary = format(value, 'b').zfill(8)
    binary = list(binary)

    return np.array(map(int, binary))

def bin2int(array):
    string = ''.join(map(str, array))

    return int(string, 2)

def restore_image(img, points, mlp):
    pixels = img.load()

    for x, y in points:
        inputs = np.array([x, y])
        output = mlp.test(inputs, discretize=True)
        gray = bin2int(output)

        pixels[x, y] = gray

    return img

if __name__ == '__main__':
    mlp = MLP((2, 10, 10, 8)) # create MLP
    img = Image.open(IMAGE_PATH).convert('L') # convert to grayscale

    # show original image
    img.show()

    width, height = img.size

    # get image pixels positions
    points = []
    for i in range(width):
        for j in range(height):
            points.append([i, j])

    # ---------
    # add noise
    # ---------
    random.shuffle(points)

    noise = (width * height) * NOISE_LEVEL
    noise = int(noise)

    noise_points = points[:noise]
    training_points = points[noise:]

    add_noise(img, noise_points)

    # show image with noise
    img.show()

    # ----------
    # training
    # ----------
    training_set = generate_training_set(img, training_points)

    converged, epochs = mlp.train(
        training_set,
        learning_rate=0.1,
        max_epochs=10000,
        min_error=0.01)

    if converged:
        print u'La red convergió en {}'.format(epochs)
    else:
        print u'La red no convergió'

    # ----------
    # restoration
    # ----------
    restore_image(img, noise_points, mlp)

    # show restored image
    img.show()
