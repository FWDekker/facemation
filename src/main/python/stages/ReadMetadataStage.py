import sys
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Pipeline import Frame, Stage


class ReadMetadataStage(Stage):
    """
    Reads simple image metadata.
    """

    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        For each frame in [frames], writes the input image's hash into key `"hash"` and pixel dimensions into key
        `"dims"`.

        :param frames: the frames to process
        :return: the processed frames
        """

        for frame in tqdm(frames, desc="Reading image metadata", file=sys.stdout):
            frame["hash"] = Hasher.hash_file(frame["path"])

            img = Image.open(frame["path"])
            width, height = img.size
            exif = img.getexif().get(0x0112)
            if exif == 6 or exif == 8:
                width, height = height, width  # Swap if EXIF tag indicates 90-degree rotation
            frame["dims"] = np.array([width, height])

        return frames
