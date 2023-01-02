import copy
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Sequence

from mergedeep import merge

import Files
from Cache import ImageCache
from UserException import UserException

"""The data read about the image during the pre-processing stage."""
ImageData = Dict[str, Any]
"""Describes the images to process, as a mapping from the image's path to the pre-processing data."""
Images = Dict[str, ImageData]


class Stage(ABC):
    """
    A part of the [Pipeline].
    """

    # TODO: Move validation to start, to catch errors early and not halfway through!
    pass


class PreprocessingStage(Stage):
    """
    Performs calculations based exclusively on the input images, to be used by later processing steps.
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
    Processes input images and stores results.
    """

    @abstractmethod
    def process(self, imgs: Images) -> ImageCache:
        """
        Processes the images in [imgs] and produces new images, stored in the returned cache under the hash of the input
        image.

        :param imgs: a read-only mapping from input image paths to their pre-processed data
        :return: the cache containing the processed images, stored under the hash of the input image
        """

        pass


class PostprocessingStage(Stage):
    """
    Performs post-processing on the produced images so far.
    """

    @abstractmethod
    def postprocess(self, imgs: Images, input_cache: ImageCache) -> ImageCache:
        """
        Post-processes the images in [input_cache] identified by [imgs].

        :param imgs: the data of the original input images
        :param input_cache: the cache containing the processed images to post-process
        :return: the cache containing the post-processed images, stored under the hash of the original input image
        """

        pass


class Pipeline:
    """
    A pipeline describing a sequential three-phase processing of input images.
    """

    """The pre-processing stages of the pipeline."""
    preprocessing: List[PreprocessingStage]
    """The processing stage of the pipeline."""
    processing: ProcessingStage or None
    """The post-processing stage of the pipeline."""
    postprocessing: List[PostprocessingStage]

    def __init__(self):
        """
        Constructs a new [Pipeline].
        """

        self.preprocessing = []
        self.processing = None
        self.postprocessing = []

    def register(self, stages: Sequence[Stage]) -> None:
        """
        Registers the [Stage]s to execute in the pipeline.

        Stages of one type are executed in the order in which they are added. Only one [ProcessingStage] can be
        registered.

        :param stages: the [Stage]s to execute in the pipeline
        :return: `None`
        """

        for stage in stages:
            if isinstance(stage, PreprocessingStage):
                self.preprocessing.append(stage)
            elif isinstance(stage, ProcessingStage):
                self.processing = stage
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

    # TODO: Merge `process` and `postprocess`, and make `postprocess` work on completed frames, and let `process`
    # TODO: ... dynamically work without an input cache (i.e. "fake cache")
    def process(self, imgs: Images) -> ImageCache:
        """
        Executes the [self.processing] stage and returns the cache containing the outputs.

        :param imgs: the images to process
        :return: the cache containing the outputs of the [self.processing] stage
        """

        if self.processing is None:
            raise Exception

        return self.processing.process(copy.deepcopy(imgs))

    def postprocess(self, imgs: Images, input_cache: ImageCache) -> None:
        """
        Executes each of the [self.postprocessing] stages in sequence on each other's outputs, starting with the outputs
        in [input_cache].

        :param imgs: the data of the original input images
        :param input_cache: the cache containing the processed images to post-process
        :return: `None`
        """

        for stage in self.postprocessing:
            input_cache = stage.postprocess(copy.deepcopy(imgs), input_cache)

    def run(self, input_dir: str) -> None:
        """
        Runs the pipeline from start to end.

        Raises a [UserException] if no supported images are found in [input_dir].

        :param input_dir: the directory containing the images to process
        :return: `None`
        """

        imgs = self.preprocess(input_dir)
        cache = self.process(imgs)
        self.postprocess(imgs, cache)
