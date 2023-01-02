import copy
import math
import sys
from pathlib import Path
from typing import Callable, Any, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Cache import ImageCache
from Pipeline import ImageInfo, ProcessingStage


class CaptionStage(ProcessingStage):
    """
    Adds a caption to each image.
    """

    captioned_cache: ImageCache
    caption_generator: Callable[[str, Any], str]

    def __init__(self, cache_dir: str, caption_generator: Callable[[str, Image], str]):
        """
        Constructs a new `CaptionStage`.

        :param cache_dir: the directory to cache captioned images in
        :param caption_generator: generates a caption based on the filename and PIL `Image` object
        """

        self.captioned_cache = ImageCache(cache_dir, "captioned", ".jpg")
        self.caption_generator = caption_generator

    def process(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Captions all [imgs] using [self.caption_generator], storing the results in [self.captioned_cache].

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed input path
        :return: a copy of [imgs] with `"processed_path"` pointing to the newly processed images
        """

        processed_imgs = copy.deepcopy(imgs)

        for img_path, img_data in tqdm(imgs.items(), desc="Adding captions", file=sys.stdout):
            caption = self.caption_generator(img_path.name, Image.open(img_path))

            processed_img_hash = Hasher.hash_file(img_data["processed_path"])
            state_hash = Hasher.hash_string(f"{processed_img_hash}-{caption}")
            if self.captioned_cache.has(img_data["hash"], state_hash):
                processed_imgs[img_path]["processed_path"] = self.captioned_cache.path(img_data["hash"], state_hash)
                continue

            img = cv2.imread(str(img_data["processed_path"]))
            img = write_on_image(img, caption, (0.05, 0.95), 0.05)

            processed_imgs[img_path]["processed_path"] = self.captioned_cache.cache(img, img_data["hash"], state_hash)

        return processed_imgs


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
    text_pos = math.floor(pos[0] * width), math.floor(pos[1] * height)

    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(0, 0, 0), thickness=32, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(255, 255, 255), thickness=16, lineType=cv2.LINE_AA)
    return image
