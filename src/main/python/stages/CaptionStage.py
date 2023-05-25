import sys
from pathlib import Path
from typing import Callable, List, TypedDict

import numpy as np
from PIL import ImageDraw, ImageFont, Image
from tqdm import tqdm

import Hasher
import Resolver
from Cache import ImageCache
from Pipeline import Frame, Stage

CaptionGenerator = Callable[[str], str]
CaptionConfig = TypedDict("CaptionConfig", {"enabled": bool,
                                            "generator": CaptionGenerator})


class CaptionStage(Stage):
    """
    Adds a layer with captions for each frame.
    """

    layer_out: int
    captioned_cache: ImageCache
    caption_generator: CaptionGenerator

    def __init__(self, layer_out: int, cache_dir: str, caption_generator: CaptionGenerator):
        """
        Constructs a new `CaptionStage`.

        :param layer_out: the layer to write caption images into
        :param cache_dir: the directory to cache captioned images in
        :param caption_generator: generates a caption based on the filename
        """

        self.layer_out = layer_out
        self.captioned_cache = ImageCache(cache_dir, "caption", ".png")
        self.caption_generator = caption_generator

    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        Captions all [frames] using [self.caption_generator], storing the results in [self.captioned_cache].

        :param frames: the frames to caption
        :return: the framed captions
        """

        for frame in tqdm(frames, desc="Adding captions", file=sys.stdout):
            caption = str(self.caption_generator(Path(frame["path"]).name))

            caption_hash = Hasher.hash_string(caption)
            state_hash = Hasher.hash_string(np.array2string(frame["dims"]))
            if self.captioned_cache.has(caption_hash, state_hash):
                frame["layers"][self.layer_out] = self.captioned_cache.path(caption_hash, state_hash)
                continue

            width, height = frame["dims"]
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pos = (0.05 * width, 0.90 * height)
            font = ImageFont.truetype(str(Resolver.resource_path("Roboto-Regular.ttf")), int(0.05 * height))

            img_draw = ImageDraw.Draw(img)
            img_draw.text(pos, caption, font=font, stroke_fill=(0, 0, 0), stroke_width=8)
            img_draw.text(pos, caption, font=font, stroke_fill=(255, 255, 255), stroke_width=1)

            frame["layers"][self.layer_out] = self.captioned_cache.cache(img, caption_hash, state_hash)

        return frames
