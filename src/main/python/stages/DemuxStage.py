import os
import shutil
import subprocess
import sys

from natsort import natsorted
from tqdm import tqdm

import Files
from Cache import ImageCache
from Config import FacemationConfig
from Pipeline import PostprocessingStage, Images
from UserException import UserException


class DemuxStage(PostprocessingStage):
    """
    Demuxes the processed frames into a video.
    """

    """The directory to store the processed frames in."""
    frames_dir: str
    """The directory to store the video in."""
    output_path: str
    """The configuration for the demuxer."""
    cfg: FacemationConfig

    def __init__(self, frames_dir: str, output_path: str, cfg: FacemationConfig):
        """
        Constructs a new [DemuxStage].

        Raises a [UserException] if this stage is enabled but FFmpeg is not installed.

        :param frames_dir: the directory to store the processed frames in
        :param output_path: the directory to store the video in
        :param cfg: the configuration for the demuxer
        """

        Files.cleardir(frames_dir)
        Files.rm(output_path)

        self.frames_dir = frames_dir
        self.output_path = output_path
        self.cfg = cfg

        # Validate requirements and inputs
        if cfg["enabled"] and shutil.which("ffmpeg") is None:
            raise UserException(f"FFmpeg is enabled in your configuration but is not installed. "
                                f"Without FFmpeg, Facemation can create frames, but cannot produce a video. "
                                f"Install FFmpeg or disable FFmpeg in your configuration. "
                                f"Check the README for more information.")

    def postprocess(self, imgs: Images, input_cache: ImageCache) -> ImageCache:
        """
        Given the original input images in [imgs], selects the corresponding processed images from [input_cache] and
        stores these in [self.frames_dir], and demuxes the contents of [self.frames_dir] into a video in
        [self.output_path] using FFmpeg.

        :param imgs: the metadata of the images from which the inputs are derived
        :param input_cache: the cache to select frames to process from
        :return: [input_cache]
        """

        if not self.cfg["enabled"]:
            return input_cache

        pbar = tqdm(natsorted(imgs.keys()), desc="Selecting frames", file=sys.stdout)
        for idx, image_path in enumerate(pbar):
            captioned_path = input_cache.get_path_any(imgs[image_path]["hash"])
            # TODO: Use hard links on Windows
            os.symlink(os.path.relpath(captioned_path, self.frames_dir), f"{self.frames_dir}/{idx}.jpg")

        print("Demuxing into video:")
        try:
            subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-stats",
                "-y",
                "-f", "image2",
                "-r", self.cfg["fps"],
                "-i", "%d.jpg",
                "-vcodec", self.cfg["codec"],
                "-crf", self.cfg["crf"],
                "-vf", ",".join(self.cfg["video_filters"]),
                self.output_path
            ], cwd=self.frames_dir, stderr=sys.stdout, check=True)
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None
