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


def get_dims(path: Path) -> (int, int):
    """
    Returns the dimensions of the image at [path] as a tuple of width and height.

    :param path: the path to the image
    :return: the dimensions of the image at [path] as a tuple of width and height
    """

    img = Image.open(path)
    width, height = img.size
    exif = img.getexif().get(0x0112)
    if exif == 6 or exif == 8:
        width, height = height, width  # Swap if EXIF tag indicates 90-degree rotation
    return width, height
