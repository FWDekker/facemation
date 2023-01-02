import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Pipeline import PreprocessingStage, Images


class ReadMetadataStage(PreprocessingStage):
    def preprocess(self, imgs: Images) -> Images:
        """
        Reads image hash and dimensions as image metadata.

        :param imgs: the paths to the input images
        :return: a mapping from file paths to the hash of the image and the dimensions of the image
        """

        metadata = {}

        for img_path in tqdm(imgs.keys(), desc="Reading image meta-data", file=sys.stdout):
            img_hash = Hasher.hash_file(img_path)

            img = Image.open(img_path)
            width, height = img.size
            exif = img.getexif().get(0x0112)
            if exif == 6 or exif == 8:
                width, height = height, width  # Swap if EXIF tag indicates 90-degree rotation
            img_dims = np.array([width, height])

            metadata[img_path] = {"hash": img_hash, "dims": img_dims}

        return metadata
