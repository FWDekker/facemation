import copy
import sys
from pathlib import Path
from typing import Callable, Dict, TypedDict

from PIL import ImageDraw, ImageFont
from tqdm import tqdm

import Hasher
import Resolver
from Cache import ImageCache
from ImageLoader import load_image
from Pipeline import ImageInfo, ProcessingStage

CaptionGenerator = Callable[[str], str]
CaptionConfig = TypedDict("CaptionConfig", {"enabled": bool,
                                            "generator": CaptionGenerator})


class CaptionStage(ProcessingStage):
    """
    Adds a caption to each image.
    """

    captioned_cache: ImageCache
    caption_generator: CaptionGenerator

    def __init__(self, cache_dir: str, caption_generator: CaptionGenerator):
        """
        Constructs a new `CaptionStage`.

        :param cache_dir: the directory to cache captioned images in
        :param caption_generator: generates a caption based on the filename
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
            caption = self.caption_generator(img_path.name)

            processed_img_hash = Hasher.hash_file(img_data["processed_path"])
            state_hash = Hasher.hash_string(f"{processed_img_hash}-{caption}")
            if self.captioned_cache.has(img_data["hash"], state_hash):
                processed_imgs[img_path]["processed_path"] = self.captioned_cache.path(img_data["hash"], state_hash)
                continue

            img = load_image(img_data["processed_path"])

            width, height = img.size
            pos = (0.05 * width, 0.90 * height)
            font = ImageFont.truetype(str(Resolver.resource_path("Roboto-Regular.ttf")), int(0.05 * height))

            img_draw = ImageDraw.Draw(img)
            img_draw.text(pos, caption, font=font, stroke_fill=(0, 0, 0), stroke_width=8)
            img_draw.text(pos, caption, font=font, stroke_fill=(255, 255, 255), stroke_width=1)

            processed_imgs[img_path]["processed_path"] = self.captioned_cache.cache(img, img_data["hash"], state_hash)

        return processed_imgs
