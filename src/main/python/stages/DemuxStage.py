import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, TypedDict, List, Union

import Files
from Pipeline import PostprocessingStage, ImageInfo
from UserException import UserException

DemuxConfig = TypedDict("DemuxConfig", {"enabled": bool,
                                        "fps": Union[str, int],
                                        "codec": str,
                                        "crf": Union[str, int],
                                        "video_filters": List[str]})


class DemuxStage(PostprocessingStage):
    """
    Demuxes the processed frames into a video.
    """

    output_path: str
    cfg: DemuxConfig

    def __init__(self, output_path: str, cfg: DemuxConfig):
        """
        Constructs a new `DemuxStage`.

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

    def postprocess(self, imgs: Dict[Path, ImageInfo], frames_dir: str) -> None:
        """
        Demuxes the images in [frames_dir] into a video in [self.output_path] using FFmpeg, subject to [self.cfg].

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed output
        path
        :param frames_dir: the directory containing exactly all processed images
        :return: `None`
        """

        args = ["ffmpeg"]
        args += ["-hide_banner"]
        args += ["-loglevel", "error"]
        args += ["-stats"]
        args += ["-y"]
        args += ["-f", "image2"]
        args += ["-r", str(self.cfg["fps"])]
        args += ["-i", "%d.jpg"]
        args += ["-vcodec", self.cfg["codec"]]
        args += ["-crf", str(self.cfg["crf"])]
        if len(self.cfg["video_filters"]) > 0:
            args += ["-vf", ",".join(self.cfg["video_filters"])]
        args += [os.path.relpath(self.output_path, frames_dir)]

        print("Demuxing into video:")
        try:
            subprocess.run(args, cwd=frames_dir, stderr=sys.stdout, check=True)
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None
