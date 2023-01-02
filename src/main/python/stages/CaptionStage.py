import math
import os
import sys
from typing import Dict, Callable, Any

import cv2
import numpy as np
from PIL import Image
from numpy import ndarray
from tqdm import tqdm

import Hasher
from Cache import ImageCache
from Pipeline import PostprocessingStage, Images


class CaptionStage(PostprocessingStage):
    """Adds a caption to each image."""

    """The cache to store captioned images in."""
    captioned_cache: ImageCache
    """Generates a caption based on the filename and PIL `Image` object."""
    caption_generator: Callable[[str, Any], str]

    def __init__(self, cache_dir: str, caption_generator: Callable[[str, Image], str]):
        """
        Constructs a new [CaptionStage].

        :param cache_dir: the directory to cache captioned images in
        :param caption_generator: generates a caption based on the filename and PIL `Image` object
        """

        self.captioned_cache = ImageCache(cache_dir, "captioned", ".jpg")
        self.caption_generator = caption_generator

    def postprocess(self, imgs: Images, input_cache: ImageCache) -> ImageCache:
        """
        For each image in [imgs], finds the corresponding image in [input_cache], and adds a caption using
        [self.caption_generator], storing the captioned images in [self.captioned_cache].

        :param imgs: the metadata of the images from which the normalized inputs are derived
        :param input_cache: the cache to read the images to caption from, with keys matching those in [imgs]
        :return: [self.captioned_cache]
        """

        pbar = tqdm(imgs.items(), desc="Adding captions", file=sys.stdout)
        for img_path, img_data in pbar:
            caption = self.caption_generator(os.path.basename(img_path), Image.open(img_path))

            img_hash = input_cache.get_path_any(img_data["hash"])
            args_hash = Hasher.hash_string(f"{img_hash}{caption}")
            if self.captioned_cache.has(img_data["hash"], args_hash):
                continue

            img = input_cache.load_any(img_data["hash"])
            img = write_on_image(img, caption, (0.05, 0.95), 0.05)
            self.captioned_cache.cache(img_data["hash"], args_hash, img)

        return self.captioned_cache


def write_on_image(image: np.ndarray, text: str, pos: [float, float], text_height: float) -> np.ndarray:
    """
    Writes [text] on [image] at coordinates [pos] with a height of [text_height].

    :param image: the image to write text on; this image is not modified
    :param text: the text to write onto [image]
    :param pos: the coordinates to place the text at, as a ratio of the size of [image]
    :param text_height: the height of the text, as a ratio of the height of [image]
    :return: a copy of [image] with text written on it
    """

    height, width = image.shape[:2]
    text_scale = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=32)
    text_scale = text_height / (text_scale[0][1] / height)
    text_pos = (math.floor(pos[0] * width), math.floor(pos[1] * height))

    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(0, 0, 0), thickness=32, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(255, 255, 255), thickness=16, lineType=cv2.LINE_AA)
    return image
