import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Pipeline import PreprocessingStage, ImageInfo


class ReadMetadataStage(PreprocessingStage):
    """
    Reads simple image metadata.
    """

    def preprocess(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Reads image hash and dimensions as image metadata.

        :param imgs: a read-only mapping from original input paths to the preprocessed data obtained thus far
        :return: a mapping from original input path to the hash and dimensions of the image
        """

        metadata = {}

        for img_path in tqdm(imgs.keys(), desc="Reading image metadata", file=sys.stdout):
            img_hash = Hasher.hash_file(img_path)

            img = Image.open(img_path)
            width, height = img.size
            exif = img.getexif().get(0x0112)
            if exif == 6 or exif == 8:
                width, height = height, width  # Swap if EXIF tag indicates 90-degree rotation
            img_dims = np.array([width, height])

            metadata[img_path] = {"hash": img_hash, "dims": img_dims}

        return metadata
