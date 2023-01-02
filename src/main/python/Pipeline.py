import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from mergedeep import merge
from natsort import natsorted

import Files
from UserException import UserException

"""The data read about the image during the pre-processing stage."""
ImageData = Dict[str, Any]
"""Describes the images to process, as a mapping from the image's path to the pre-processing data."""
Images = Dict[str, ImageData]


class Stage(ABC):
    """
    A part of the [Pipeline].
    """


class PreprocessingStage(Stage):
    """
    Performs calculations based exclusively on the input images, to be used by later processing steps.

    Preprocessing stages can depend on each other.
    """

    @abstractmethod
    def preprocess(self, imgs: Images) -> Images:
        """
        Preprocesses the images in [imgs] and returns some data.

        :param imgs: a read-only mapping from paths to the data obtained thus far
        :return: the new data calculated by this stage
        """

        pass


class ProcessingStage(Stage):
    """
    Processes images into new images.

    Processing stages are typically chained together.
    """

    @abstractmethod
    def process(self, imgs: Images, input_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Given the original input images [imgs], continues the processing pipeline from the (partially) processed images
        in [input_paths], storing the results in the returned paths.

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :param input_paths: a read-only mapping from input image paths to (partially) processed image paths
        :return: a mapping from input image paths to processed image paths
        """

        pass


class PostprocessingStage(Stage):
    """
    Turns processed images into some desired output.

    Postprocessing stages do not chain information to each other.
    """

    @abstractmethod
    def postprocess(self, imgs: Images, frames_dir: str) -> None:
        """
        Post-processes the images in [input_cache] identified by [imgs].

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :param frames_dir: the directory containing processed images
        :return: `None`
        """

        pass


class Pipeline:
    """
    A pipeline describing a sequential three-phase processing of input images.
    """

    """The pre-processing stages of the pipeline."""
    preprocessing: List[PreprocessingStage]
    """The processing stage of the pipeline."""
    processing: List[ProcessingStage]
    """The post-processing stage of the pipeline."""
    postprocessing: List[PostprocessingStage]

    def __init__(self):
        """
        Constructs a new `Pipeline`.
        """

        self.preprocessing = []
        self.processing = []
        self.postprocessing = []

    def register(self, stage: Stage) -> None:
        """
        Registers [stage] as part of the pipeline.

        Stages of one type are executed in the order in which they are added. Only one [ProcessingStage] can be
        registered.

        :param stage: a stage to execute in the pipeline
        :return: `None`
        """

        if isinstance(stage, PreprocessingStage):
            self.preprocessing.append(stage)
        elif isinstance(stage, ProcessingStage):
            self.processing.append(stage)
        elif isinstance(stage, PostprocessingStage):
            self.postprocessing.append(stage)

    def preprocess(self, input_dir: str) -> Images:
        """
        Executes all [self.preprocessing] stages and returns the obtained data.

        Raises a [UserException] if no supported images are found in [input_dir].

        :param input_dir: the directory containing the images to pre-process
        :return: the data obtained by the [self.preprocessing] stages
        """

        Files.mkdir(input_dir)
        img_paths = Files.glob_extensions(input_dir, "bmp,dib,jpeg,jpg,jpe,jp2,png,pbm,pgm,ppm,sr,ras,tiff,tif")
        if len(img_paths) == 0:
            raise UserException(f"No images detected in '{Path(input_dir).resolve()}'. "
                                f"Are you sure you put them in the right place?", )

        imgs = dict([(it, {}) for it in img_paths])
        for stage in self.preprocessing:
            merge(imgs, stage.preprocess(copy.deepcopy(imgs)))
        return imgs

    def process(self, imgs: Images, frames_dir: str) -> None:
        """
        Executes the [self.processing] stage and returns the cache containing the outputs.

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :param frames_dir: the directory to store processed images in
        :return: `None`
        """

        input_paths = {it: it for it in imgs.keys()}
        for stage in self.processing:
            input_paths = stage.process(copy.deepcopy(imgs), copy.deepcopy(input_paths))

        Files.cleardir(frames_dir)
        for idx, image_path in enumerate(natsorted(imgs.keys())):
            os.symlink(os.path.relpath(image_path, frames_dir), f"{frames_dir}/{idx}.jpg")

    def postprocess(self, imgs: Images, frames_dir: str) -> None:
        """
        Executes each of the [self.postprocessing] stages in sequence on each other's outputs, starting with the outputs
        in [input_cache].

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :param frames_dir: the directory containing processed images
        :return: `None`
        """

        for stage in self.postprocessing:
            stage.postprocess(copy.deepcopy(imgs), frames_dir)

    def run(self, input_dir: str, frames_dir: str) -> None:
        """
        Runs the pipeline from start to end.

        Raises a [UserException] if no supported images are found in [input_dir].

        :param input_dir: the directory containing the images to process
        :param frames_dir: the directory to store processed frames in
        :return: `None`
        """

        imgs = self.preprocess(input_dir)
        self.process(imgs, frames_dir)
        self.postprocess(imgs, frames_dir)
