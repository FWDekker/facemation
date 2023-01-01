import os
import subprocess
import sys
from typing import Dict

from natsort import natsorted
from tqdm import tqdm

from Cache import ImageCache
from ConfigHelper import FacemationConfig
from ReadInputsStage import ImageMetadata
from UserException import UserException


def demux_images(imgs: Dict[str, ImageMetadata], input_cache: ImageCache, frames_dir: str, output_path: str,
                 cfg: FacemationConfig) -> None:
    """
    Given the original input image in [imgs], selects the corresponding processed images from [input_cache] and stores
    these in [frames_dir], and demuxes the contents of [frames_dir] into a video in [output_path] using FFmpeg.

    :param imgs: the metadata of the images from which the inputs are derived
    :param input_cache: the cache to select frames to process from
    :param frames_dir: the directory to store frame links in for FFmpeg
    :param output_path: the path relative to [input_dir] to save the created video as
    :param cfg: the `ffmpeg` part of the configuration; see also `config_default.py`
    :return: `None`
    """

    if not cfg["enabled"]:
        return

    pbar = tqdm(natsorted(imgs.keys()), desc="Selecting frames", file=sys.stdout)
    for idx, image_path in enumerate(pbar):
        captioned_path = input_cache.get_path_any(imgs[image_path]["hash"])
        os.symlink(os.path.relpath(captioned_path, frames_dir), f"{frames_dir}/{idx}.jpg")

    print("Demuxing into video:")
    try:
        subprocess.run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-stats",
            "-y",
            "-f", "image2",
            "-r", cfg["fps"],
            "-i", "%d.jpg",
            "-vcodec", cfg["codec"],
            "-crf", cfg["crf"],
            "-vf", ",".join(cfg["video_filters"]),
            output_path
        ], cwd=frames_dir, stderr=sys.stdout, check=True)
    except Exception as exception:
        raise UserException("FFmpeg failed to create a video. "
                            "Read the messages above for more information.", exception) from None
