import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, TypedDict, Union

import Files
from Pipeline import Frame, Stage
from UserException import UserException

FfmpegConfig = TypedDict("FfmpegConfig", {"enabled": bool,
                                          "exe_path": str,
                                          "frames_dir": str,
                                          "output_path": str,
                                          "fps": Union[str, int],
                                          "codec": str,
                                          "crf": Union[str, int],
                                          "video_filters": List[str],
                                          "custom_global_options": List[str],
                                          "custom_inputs": List[str],
                                          "custom_output_options": List[str]})


class FfmpegStage(Stage):
    """
    Demuxes the processed frames into a video.
    """

    layers_in: List[int]
    cfg: FfmpegConfig

    def __init__(self, layers_in: List[int], cfg: FfmpegConfig):
        """
        Constructs a new `FfmpegStage`.

        Raises a [UserException] if this stage is enabled but the FFmpeg executable cannot be found.

        :param cfg: the configuration for the demuxer
        """

        Files.rm(cfg["output_path"])

        self.layers_in = layers_in
        self.cfg = cfg

        if (not Path(cfg["exe_path"]).exists()) and (shutil.which("ffmpeg") is None):
            raise UserException(f"FFmpeg is enabled in your configuration but is not installed. "
                                f"Install FFmpeg or disable FFmpeg in your configuration. "
                                f"Check the README for more information.")

    def prepare_frames(self, frames: List[Frame]) -> None:
        Files.cleardir(self.cfg["frames_dir"])
        Files.mkdir(self.cfg["frames_dir"])

        for idx, frame in enumerate(frames):
            for layer_in in self.layers_in:
                img_path = frame["layers"][layer_in]
                link_path = f"{self.cfg['frames_dir']}/{idx}-{layer_in}{Path(img_path).suffix}"
                os.link(img_path, Path(link_path).resolve())

    def create_args(self) -> List[str]:
        args = [self.cfg["exe_path"]]
        args += ["-hide_banner"]
        args += ["-loglevel", "error"]
        args += ["-stats"]
        args += ["-y"]
        args += ["-f", "image2"]
        args += ["-r", str(self.cfg["fps"])]
        args += self.cfg["custom_global_options"]
        for layer_in in self.layers_in:
            # TODO: Detect extension better!
            extension = Path(glob.glob(f"{self.cfg['frames_dir']}/*-{layer_in}.*")[0]).suffix
            args += ["-i", f"{self.cfg['frames_dir']}/%d-{layer_in}{extension}"]
            args += self.cfg["custom_inputs"]
        args += ["-vcodec", self.cfg["codec"]]
        args += ["-crf", str(self.cfg["crf"])]
        if len(self.cfg["video_filters"]) > 0:
            args += ["-vf", ",".join(self.cfg["video_filters"])]
        args += self.cfg["custom_output_options"]
        args += [self.cfg["output_path"]]

        return args

    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        Demuxes layers [self.layers_in] of all [frames] into a video using FFmpeg, subject to [self.cfg].

        :param frames: the frames to demux
        :return: [frames], unmodified
        """

        print("Combining frames into a video:")
        self.prepare_frames(frames)
        args = self.create_args()

        try:
            subprocess.run(args, stderr=sys.stdout, check=True)

            print(f"Your video has been saved in {Path(self.cfg['output_path']).resolve()}.")
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None

        return frames
