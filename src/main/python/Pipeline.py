from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import PIL
from PIL import Image
from natsort import natsorted

import Files
from UserException import UserException

PipelineConfig = TypedDict("PipelineConfig", {"input_dir": str, "cache_dir": str})

# `Frame` contains (meta)data for an image. Various stages add, remove, and modify the contents of a `Frame`. However,
# the following keys always exist:
# * `"path"` is the original input image's absolute path
# * `"layers"` is a dictionary of the layer name to the absolute path of the corresponding image
# A `Frame` typically does not contain images directly, but only paths to the images. When a stage alters an image, it
# will typically write the altered image into a new file, and point the path in the `Frame` to this new file.
Frame = Dict[str, Any]


class Stage(ABC):
    """
    Processes the list of frames.

    Stages can perform any arbitrary processing, including any combination of adding or removing metadata, adding or
    dropping frames, adding or dropping layers, and so on.

    Stages can be linked together by specifying in- and output layers in the stage's constructor. By default, a stage
    reads from and writes to layer `0`.
    """

    @abstractmethod
    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        Processes the [frames] in some way.

        :param frames: the frames to process; may be modified
        :return: the processed frame; may be the same instance as [frames]
        """

        pass


class Pipeline:
    """
    A sequence of [Stage]s to process on a specified directory.
    """

    stages: List[Stage]

    def __init__(self):
        """
        Constructs a new `Pipeline`.
        """

        self.stages = []

    def register(self, stage: Stage) -> None:
        """
        Registers [stage] as part of the pipeline.

        Stages of one type are executed in the order in which they are added.

        :param stage: a stage to execute in the pipeline
        :return: `None`
        """

        self.stages.append(stage)

    @staticmethod
    def read_dir(input_dir: str) -> List[Frame]:
        """
        Reads the input directory and returns the list of `Frame`s to process.

        Raises a [UserException] if [input_dir] is empty, or if [input_dir] contains an unsupported image.

        :param input_dir: the directory containing the images to process
        :return: `None`
        """

        Files.mkdir(input_dir)
        img_paths = [str(it.resolve()) for it in Path(input_dir).iterdir() if it.is_file()]

        if len(img_paths) == 0:
            raise UserException(f"No images detected in '{Path(input_dir).resolve()}'. "
                                f"Are you sure you put them in the right place?")

        for it in img_paths:
            try:
                Image.open(it)
            except PIL.UnidentifiedImageError:
                raise UserException(f"Unsupported image type for input '{it}'.")

        return [{"path": it, "layers": {0: it}} for it in natsorted(img_paths)]

    def run(self, input_dir: str) -> None:
        """
        Runs the pipeline from start to end.

        Raises a [UserException] if [input_dir] is empty, if [input_dir] contains an unsupported image, or if any stage
        raises a [UserException].

        :param input_dir: the directory containing the images to process
        :return: `None`
        """

        frames = self.read_dir(input_dir)
        for stage in self.stages:
            frames = stage.process(frames)
