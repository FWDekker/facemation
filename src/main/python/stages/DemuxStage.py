import shutil
import subprocess
import sys

import Files
from Config import FacemationConfig
from Pipeline import PostprocessingStage, Images
from UserException import UserException


class DemuxStage(PostprocessingStage):
    """
    Demuxes the processed frames into a video.
    """

    """The directory to store the video in."""
    output_path: str
    """The configuration for the demuxer."""
    cfg: FacemationConfig

    def __init__(self, output_path: str, cfg: FacemationConfig):
        """
        Constructs a new [DemuxStage].

        Raises a [UserException] if this stage is enabled but FFmpeg is not installed.

        :param output_path: the directory to store the video in
        :param cfg: the configuration for the demuxer
        """

        Files.rm(output_path)

        self.output_path = output_path
        self.cfg = cfg

        if shutil.which("ffmpeg") is None:
            raise UserException(f"FFmpeg is enabled in your configuration but is not installed. "
                                f"Without FFmpeg, Facemation can create frames, but cannot produce a video. "
                                f"Install FFmpeg or disable FFmpeg in your configuration. "
                                f"Check the README for more information.")

    def postprocess(self, imgs: Images, frames_dir: str) -> None:
        """
        Demuxes the images in [frames_dir] into a video in [self.output_path] using FFmpeg, subject to [self.cfg].

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :param frames_dir: the directory containing processed images
        :return: `None`
        """

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
            ], cwd=frames_dir, stderr=sys.stdout, check=True)
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None
