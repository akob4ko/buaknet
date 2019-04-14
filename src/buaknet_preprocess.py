import base64
from io import BytesIO
from PIL import Image
import time
import numpy as np


def print_elapsed_time(_time):
    elapsed_time = time.time() - _time
    if elapsed_time < 60:
        print("{:2.1f} sec.".format(elapsed_time))
    elif 60 < elapsed_time < 3600:
        print("{:2.1f} min.".format(elapsed_time / 60))
    else:
        print("{:2.1f} hr.".format(elapsed_time / 3600))


def base64_str_to_numpy(base64_str):
    base64_str = str(base64_str)
    if "base64" in base64_str:
        _, base64_str = base64_str.split(',')
    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    buf.seek(0)
    pimg = Image.open(buf)
    img = np.array(pimg)
    return img[:, :, 3]


def resize_img(img_ndarray):
    img = Image.fromarray(img_ndarray)
    img.thumbnail((28, 28), Image.ANTIALIAS)
    return np.array(img)


def reshape_array(img_ndarray):
    img_std = img_ndarray / 255
    digit = np.reshape(img_std, (784, 1))
    return digit
