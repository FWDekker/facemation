import warnings
from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: Path) -> Image:
    """
    Loads the image at [path], rotated if necessary, without throwing annoying warnings.

    :param path: the path to the image to load
    :return: the image at [path], rotated if necessary
    """

    img = Image.open(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = ImageOps.exif_transpose(img)

    return img

