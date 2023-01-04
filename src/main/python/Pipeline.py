import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, TypedDict

import PIL
from PIL import Image
from mergedeep import merge
from natsort import natsorted

import Files
from UserException import UserException

PipelineConfig = TypedDict("PipelineConfig", {"input_dir": str, "cache_dir": str, "frames_dir": str})
ImageInfo = Dict[str, Any]  # Contains `processed_path` during processing, and `frame_path` during postprocessing


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
    def preprocess(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Preprocesses the images in [imgs] and returns new data.

        :param imgs: a read-only mapping from original input paths to the preprocessed data obtained thus far
        :return: a mapping from original input path to the new data, to be merged into [imgs]
        """

        pass


class ProcessingStage(Stage):
    """
    Processes images into new images.

    Processing stages are typically chained together.
    """

    @abstractmethod
    def process(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Processes the images in [imgs] into new images.

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed input path
        :return: a copy of [imgs] with `"processed_path"` pointing to the newly processed images
        """

        pass


class PostprocessingStage(Stage):
    """
    Turns processed images into some desired output.

    Postprocessing stages do not chain information to each other.
    """

    @abstractmethod
    def postprocess(self, imgs: Dict[Path, ImageInfo], frames_dir: str) -> None:
        """
        Postprocesses the images in [imgs] into some desired output.

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed output
        path
        :param frames_dir: the directory containing exactly all processed images
        :return: `None`
        """

        pass


class Pipeline:
    """
    A pipeline describing a sequential three-phase processing of input images.
    """

    preprocessing: List[PreprocessingStage]
    processing: List[ProcessingStage]
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

        Stages of one type are executed in the order in which they are added.

        :param stage: a stage to execute in the pipeline
        :return: `None`
        """

        if isinstance(stage, PreprocessingStage):
            self.preprocessing.append(stage)
        elif isinstance(stage, ProcessingStage):
            self.processing.append(stage)
        elif isinstance(stage, PostprocessingStage):
            self.postprocessing.append(stage)

    def preprocess(self, input_dir: str) -> Dict[Path, ImageInfo]:
        """
        Executes all [self.preprocessing] stages and returns the obtained data.

        Raises a [UserException] if no supported images are found in [input_dir].

        :param input_dir: the directory containing the images to preprocess
        :return: a mapping from original input path to the preprocessed data
        """

        Files.mkdir(input_dir)

        img_paths = [it for it in Path(input_dir).iterdir() if it.is_file()]
        if len(img_paths) == 0:
            raise UserException(f"No images detected in '{Path(input_dir).resolve()}'. "
                                f"Are you sure you put them in the right place?", )
        for img_path in img_paths:
            try:
                Image.open(img_path)
            except PIL.UnidentifiedImageError:
                raise UserException(f"Unsupported image type for input '{img_path}'.")

        imgs = {Path(it): {} for it in img_paths}
        for stage in self.preprocessing:
            merge(imgs, stage.preprocess(copy.deepcopy(imgs)))
        return imgs

    def process(self, imgs: Dict[Path, ImageInfo], frames_dir: str) -> Dict[Path, ImageInfo]:
        """
        Executes the [self.processing] stage and returns the cache containing the outputs.

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed input path
        :param frames_dir: the directory to store processed images in
        :return: a copy of [imgs] with `"processed_path"` pointing to the newly processed images
        """

        Files.cleardir(frames_dir)

        processed_imgs = copy.deepcopy(imgs)
        for img_path in list(processed_imgs.keys()):
            processed_imgs[img_path]["processed_path"] = img_path

        for stage in self.processing:
            processed_imgs = stage.process(copy.deepcopy(processed_imgs))

        for idx, img_path in enumerate(natsorted(processed_imgs.keys())):
            frame_path = f"{frames_dir}/{idx}.jpg"
            os.link(processed_imgs[img_path]["processed_path"].resolve(), frame_path)
            processed_imgs[img_path]["frame_path"] = frame_path

        return processed_imgs

    def postprocess(self, imgs: Dict[Path, ImageInfo], frames_dir: str) -> None:
        """
        Executes each of the [self.postprocessing] stages in sequence on each other's outputs, starting with the outputs
        in [input_cache].

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed output
        path
        :param frames_dir: the directory containing exactly all processed images
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
        imgs = self.process(imgs, frames_dir)
        self.postprocess(imgs, frames_dir)
