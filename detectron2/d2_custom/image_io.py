import os.path as osp

import cv2
import numpy as np
from PIL import Image

import six

from detectron2.d2_custom.zipreader import ZipReader

from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED


imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def is_zip_path(img_or_path):
    return '.zip@' in img_or_path


def imread(img_or_path, flag='color'):
    """Read an image.

    Args:
        img_or_path (ndarray or str): Either a numpy array or image path.
            If it is a numpy array (loaded image), then it will be returned
            as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        flag = imread_flags[flag] if is_str(flag) else flag
        if is_zip_path(img_or_path):
            img = imfrombytes(ZipReader.read(img_or_path), flag)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            return img
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        img = cv2.imread(img_or_path, flag)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img
    else:
        # flag = imread_flags[flag] if is_str(flag) else flag
        # img1 = Image.open(img_or_path).convert("RGB")
        # img_np1 = np.asarray(img1)
        #
        # img2 = cv2.imread(img_or_path.name, flag)
        # img_np2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # print(img_np1 - img_np2)
        # print(np.abs(img_np1 - img_np2).max())
        # print((np.abs(img_np1 - img_np2) > 0).sum() / (img_np1.shape[0] * img_np1.shape[1]))
        # print(img_np.shape)

        return Image.open(img_or_path)
        # raise TypeError('"img" must be a numpy array or a filename')


def imfrombytes(content, flag='color'):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)

    return img
