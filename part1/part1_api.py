try:
    import os
    import json
    import glob
    import argparse
    import math

    from skimage.feature import *
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

"""
:return the coordinates of the the suspicious lights.
"""


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    the main function that finds the traffic lights.
    :return: the coordinates of the red and green traffic lights.
    """
    c_image = c_image.astype(float)
    x_red, y_red = find_lights(c_image, 0)
    x_green, y_green = find_lights(c_image, 1)
    # remove duplicates coordinates, between the red and green
    for i in range(len(x_red)):
        min_distance = 2000
        index = 0
        j = 0
        while j < len(x_green):
            distance = math.sqrt(((x_red[i] - x_green[j]) ** 2) + ((y_red[i] - y_green[j]) ** 2))
            if distance <= min_distance:
                min_distance = distance
                index = j
            j += 1
        if min_distance <= 10:
            x_green = x_green[:index] + x_green[index + 1:]
            y_green = y_green[:index] + y_green[index + 1:]
    candidates = [[x, y] for x, y in zip(x_red, y_red)]
    auxiliary = [1] * len(x_red)
    candidates += [[x, y] for x, y, in zip(x_green, y_green)]
    auxiliary += [0] * len(x_green)
    return candidates, auxiliary


# Find the lights using 2 drawn pictures in two sizes of traffic_lights as kernels.
def find_lights(c_image, color_num):
    c_image = c_image[:, :, color_num]
    x_light, y_light = [], []
    x_light, y_light = get_coordinates(c_image, x_light, y_light, 'traffic_lights.png')
    x_light, y_light = get_coordinates(c_image, x_light, y_light, 'small_traffic_lights.png')
    return x_light, y_light


# Returns the coordinates of the suspicious lights
def get_coordinates(c_image, x_light, y_light, image):
    returned_x_light, returned_y_light = convolve_picture(c_image, image)
    x_light += returned_x_light
    y_light += returned_y_light
    return x_light, y_light


def convolve_picture(c_image, kernel):
    x_coordinates, y_coordinates = [], []
    im1 = Image.open(('part1/' + kernel)).convert('L')
    kernel = np.stack((im1,) * 3, axis=-1)
    kernel = kernel[:, :, 0].astype(float)
    # Normalize kernel
    kernel = kernel - np.average(kernel)
    kernel = kernel / np.amax(kernel)
    # Make convolution on the image
    convolved = sg.convolve(c_image, kernel, 'same')
    # Get one point that is the maximum from each area
    coordinates = peak_local_max(convolved, min_distance=10, num_peaks=10)
    # Take only coordinates that meet a certain condition
    for i in range(len(coordinates)):
        if convolved[coordinates[i][0]][coordinates[i][1]] > np.amax(convolved) - 30000 and coordinates[i][0] > 20:
            x_coordinates += [coordinates[i][1]]
            y_coordinates += [coordinates[i][0]]
    return x_coordinates, y_coordinates


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    return red_x, red_y, green_x, green_y


if __name__ == '__main__':
    print(test_find_tfl_lights('berlin_000000_000019_leftImg8bit.png'))
