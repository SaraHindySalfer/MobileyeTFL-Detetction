import numpy as np
from PIL import Image


def crop_image(path, x, y):
    im = Image.open(path)
    im = im.crop((x, y, x + 81, y + 81))
    cropped_image = np.asarray(im, dtype='uint8')
    return cropped_image
