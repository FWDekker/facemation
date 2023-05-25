import math
import sys
import warnings
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Cache import ImageCache
from ImageLoader import load_image
from Pipeline import Frame, Stage
from UserException import UserException


class NormalizeStage(Stage):
    """
    Centers the face in each frame.
    """

    layer_in: int
    layer_out: int
    normalized_cache: ImageCache

    def __init__(self, layer_in: int, layer_out: int, cache_dir: str):
        """
        Constructs a new `NormalizeStage`.

        :param layer_in: the layer to read images to normalize from
        :param layer_out: the layer to write normalized images into
        :param cache_dir: the directory to store normalized images in
        """

        self.layer_in = layer_in
        self.layer_out = layer_out
        self.normalized_cache = ImageCache(cache_dir, "normalized", ".jpg")

    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        Translates, rotates, and resizes layer [self.layer_in] of all [frames], writing to layer [self.layer_out],
        additionally caching the results in [self.normalized_cache].

        :param frames: the frames to normalize
        :return: the normalized frames
        """

        eyes = [it["eyes"] for it in frames]

        # Find scale for resizing
        eye_dists = [math.dist(it[0], it[1]) for it in eyes]
        min_eye_dist = np.min(np.array(eye_dists))
        scales = [min_eye_dist / it for it in eye_dists]
        scaled_img_dims = [(scales[idx] * frames[idx]["dims"]).astype(int) for idx in range(len(frames))]

        # Find translation to align eyes
        eye_centers = [np.mean([it[0], it[1]], axis=0).astype(int) for it in eyes]
        scaled_eye_centers = [(scales[idx] * eye_centers[idx]).astype(int) for idx in range(len(frames))]
        max_scaled_eye_center = np.max(np.array(scaled_eye_centers), axis=0)
        translations = [max_scaled_eye_center - it for it in scaled_eye_centers]

        # Find rotation angle
        # Note that angle is negated because y-axis is flipped by OpenCV, so positive angle is clockwise rotation
        scaled_relative_right_eye_positions = \
            [scales[idx] * eyes[idx][1] - scaled_eye_centers[idx] for idx in range(len(frames))]
        angles = [-math.atan2(it[1], it[0]) for it in scaled_relative_right_eye_positions]

        # Find cropping boundaries
        img_corners_after_rotation = [rotate(max_scaled_eye_center,
                                             translations[idx] + get_corners(scaled_img_dims[idx]),
                                             angles[idx]) for idx in range(len(frames))]
        img_inner_boundaries = [largest_inner_rectangle(img_corners_after_rotation[it]) for it in range(len(frames))]
        min_inner_boundaries = rectangle_overlap(np.array(img_inner_boundaries))
        min_inner_boundaries = (np.floor(min_inner_boundaries / 2) * 2).astype(int)

        # Perform normalization
        pbar = enumerate(tqdm(frames, desc="Normalizing images", file=sys.stdout))
        for idx, frame in pbar:
            eyes_string = np.array2string(eyes[idx])
            params_string = np.array2string(np.hstack([scales[idx],
                                                       translations[idx],
                                                       angles[idx],
                                                       min_inner_boundaries]))

            # Skip if cached
            state_hash = Hasher.hash_string(f"{eyes_string}-{params_string}")
            if self.normalized_cache.has(frame["hash"], state_hash):
                frame["layers"][self.layer_out] = self.normalized_cache.path(frame["hash"], state_hash)
                continue

            # Validate normalization parameters
            angle_abs = math.fabs(math.degrees(angles[idx]))
            if angle_abs >= 45.0:
                raise UserException(f"Image '{frame['path']}' is rotated by {angle_abs} degrees, but Facemation only "
                                    f"supports angles up to 45 degrees (but preferably much lower). "
                                    f"You should manually rotate the image and crop out the relevant parts, or remove "
                                    f"the image from the inputs altogether.")
            if angle_abs >= 30.0:
                warnings.warn(f"Image '{frame['path']}' is rotated by {angle_abs} degrees, which may cause a very"
                              f"small output video. "
                              f"Consider manually cropping out the relevant parts of the image, or removing the image "
                              f"from the inputs altogether.")

            # Normalize image
            translated_dims = tuple(scaled_img_dims[idx] + translations[idx])
            translation_matrix = (1, 0, -translations[idx][0], 0, 1, -translations[idx][1])

            img = load_image(frame["layers"][self.layer_in])
            img = img.resize(scaled_img_dims[idx])
            img = img.transform(translated_dims, Image.AFFINE, translation_matrix)
            img = img.rotate(-math.degrees(angles[idx]), center=tuple(max_scaled_eye_center))
            img = img.crop((min_inner_boundaries[0], min_inner_boundaries[1],
                            min_inner_boundaries[2], min_inner_boundaries[3]))

            # Store normalized image
            frame["layers"][self.layer_out] = self.normalized_cache.cache(img, frame["hash"], state_hash)

        return frames


def get_corners(dims: np.ndarray) -> np.ndarray:
    """
    Returns the corners of a rectangle at `(0, 0)` with width and height as specified in [dims].

    :param dims: the width and height of the rectangle
    :return: the corners of a rectangle at `(0, 0)` with width and height as specified in [dims]
    """

    return np.array([[dims[0], 0], [0, 0], [0, dims[1]], dims])


def rotate(origin: np.ndarray, points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates [points] around [origin] counter-clockwise by [angle] radians.

    This function does not use a flipped y-axis, so "above" is positive.

    :param origin: the point to rotate [points] around
    :param points: the points to rotate around [origin] by [angle] radians
    :param angle: the angle to rotate [points] by, in radians
    :return: the coordinates of the rotated points
    """

    cos = math.cos(angle)
    sin = math.sin(angle)
    diff = points - origin

    return np.column_stack([origin[0] + cos * diff[:, 0] - sin * diff[:, 1],
                            origin[1] + sin * diff[:, 0] + cos * diff[:, 1]]).astype(int)


def largest_inner_rectangle(corners: np.ndarray) -> np.ndarray:
    """
    Returns the largest non-rotated inner rectangle of the rectangle specified by [corners].

    :param corners: the corners of the rectangle to find the inner rectangle in
    :return: the largest non-rotated inner rectangle of the rectangle specified by [corners]
    """

    xs = np.sort(corners[:, 0])
    ys = np.sort(corners[:, 1])

    return np.array([[xs[1], ys[1]], [xs[2], ys[2]]])


def rectangle_overlap(rectangles: np.ndarray) -> np.ndarray:
    """
    Returns the largest rectangle that is within all [rectangles], assuming such a rectangle exists.

    :param rectangles: the rectangle to find the overlap of
    :return: the largest rectangle that is within all [rectangles], assuming such a rectangle exists
    """

    return np.array([np.max(rectangles[:, 0, 0]),
                     np.max(rectangles[:, 0, 1]),
                     np.min(rectangles[:, 1, 0]),
                     np.min(rectangles[:, 1, 1])])
