# -*- coding: utf-8 -*-

import numpy as np
import random
import sys
from os import path
from PIL import Image

from mlp import MultiLayerPerceptron as MLP

NOISE_LEVEL = 0.2
IMAGE_PATH = 'house.tif'

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
        # output = int2bin(gray)
        output = gray / float(255)
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
        # output = mlp.test(inputs, discretize=True)
        # gray = bin2int(output)

        gray = mlp.test(inputs, discretize=False)
        pixels[x, y] = gray * 255

    return img

if __name__ == '__main__':
    mlp = MLP((2, 28, 15, 1)) # create MLP
    img = Image.open(IMAGE_PATH).convert('L') # convert to grayscale

    file_name, file_ext = path.splitext(IMAGE_PATH)

    # show original image
    img.save(file_name + '_gray' + file_ext)

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
    img.save(file_name + '_noise' + file_ext)

    # ----------
    # training
    # ----------
    training_set = generate_training_set(img, training_points)

    converged, epochs = mlp.train(
        training_set,
        learning_rate=1.3,
        max_epochs=600,
        min_error=0.005)

    if converged:
        print u'La red convergió en ' + epochs
    else:
        print u'La red no convergió'

    # ----------
    # restoration
    # ----------

    # show restored image
    restore_image(img, noise_points, mlp)
    img.save(file_name + '_restored' + file_ext)

    # completly recreate the image using the trained MLP
    restore_image(img, points, mlp)
    img.save(file_name + '_recreated' + file_ext)
