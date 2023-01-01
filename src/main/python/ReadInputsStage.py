import glob
import sys
from typing import Dict, TypedDict

import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm

import SHA256
from Types import Dimensions

ImageMetadata = TypedDict("ImageMetadata", {"hash": str, "dims": Dimensions})


def read_image_data(input_dir: str) -> Dict[str, ImageMetadata]:
    """
    Reads image metadata, such as filesize and image hash.

    :param input_dir: the directory to read input files from
    :return: a mapping from input images to the hash of the image and the dimensions of the image
    """

    image_data = {}

    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Reading image meta-data", file=sys.stdout)
    for image_path in pbar:
        image_hash = SHA256.hash_file(image_path)

        image = Image.open(image_path)
        width, height = image.size
        exif = image.getexif().get(0x0112)
        if exif == 6 or exif == 8:
            width, height = height, width

        image_data[image_path] = {"hash": image_hash, "dims": np.array([width, height])}

    return image_data
