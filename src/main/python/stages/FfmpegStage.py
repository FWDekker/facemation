import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, TypedDict, List, Union

import Files
from Pipeline import PostprocessingStage, ImageInfo
from UserException import UserException

FfmpegConfig = TypedDict("FfmpegConfig", {"enabled": bool,
                                          "exe_path": str,
                                          "output_path": str,
                                          "fps": Union[str, int],
                                          "codec": str,
                                          "crf": Union[str, int],
                                          "video_filters": List[str],
                                          "custom_global_options": List[str],
                                          "custom_inputs": List[str],
                                          "custom_output_options": List[str]})


class FfmpegStage(PostprocessingStage):
    """
    Demuxes the processed frames into a video.
    """

    cfg: FfmpegConfig

    def __init__(self, cfg: FfmpegConfig):
        """
        Constructs a new `FfmpegStage`.

        Raises a [UserException] if this stage is enabled but the FFmpeg executable cannot be found.

        :param cfg: the configuration for the demuxer
        """

        Files.rm(cfg["output_path"])

        self.cfg = cfg

        if (not Path(cfg["exe_path"]).exists()) and (shutil.which("ffmpeg") is None):
            raise UserException(f"FFmpeg is enabled in your configuration but is not installed. "
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

        args = [self.cfg["exe_path"]]
        args += ["-hide_banner"]
        args += ["-loglevel", "error"]
        args += ["-stats"]
        args += ["-y"]
        args += ["-f", "image2"]
        args += ["-r", str(self.cfg["fps"])]
        args += self.cfg["custom_global_options"]
        args += ["-i", f"{frames_dir}/%d.jpg"]
        args += self.cfg["custom_inputs"]
        args += ["-vcodec", self.cfg["codec"]]
        args += ["-crf", str(self.cfg["crf"])]
        if len(self.cfg["video_filters"]) > 0:
            args += ["-vf", ",".join(self.cfg["video_filters"])]
        args += self.cfg["custom_output_options"]
        args += [self.cfg["output_path"]]

        print("Combining frames into a video:")
        try:
            subprocess.run(args, stderr=sys.stdout, check=True)

            print(f"Your video has been saved in {Path(self.cfg['output_path']).resolve()}.")
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None
