# -*- coding: utf-8 -*-

import numpy as np
import random
import sys

from PIL import Image

from mlp import MultiLayerPerceptron as MLP

NOISE_LEVEL = 0.2
IMAGE_PATH = 'lena100x100.png'

width = None
height = None

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

        inputs = normalize(x, y)
        output = int2bin(gray)
        training_set.append((inputs, output))

    return training_set

def normalize(x, y):
    x = x / float(width)
    y = y / float(height)

    return np.array([x, y])

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
        inputs = normalize(x, y)
        output = mlp.test(inputs, discretize=True)
        gray = bin2int(output)

        pixels[x, y] = gray

    return img

if __name__ == '__main__':
    mlp = MLP((2, 90, 50, 8)) # create MLP
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
        learning_rate=0.2,
        max_epochs=200,
        min_error=1.6)

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
