import copy
import math
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import Hasher
from Cache import ImageCache
from ImageLoader import load_image
from Pipeline import ProcessingStage, ImageInfo
from UserException import UserException


class NormalizeStage(ProcessingStage):
    """
    Normalizes input images.
    """

    normalized_cache: ImageCache

    def __init__(self, cache_dir: str):
        """
        Constructs a new `NormalizeStage`.

        :param cache_dir: the directory to store normalized images in
        """

        self.normalized_cache = ImageCache(cache_dir, "normalized", ".jpg")

    def process(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Translates, rotates, and resizes all [imgs], storing the results in [self.normalized_cache].

        :param imgs: a read-only mapping from original input paths to the preprocessed data and the processed input path
        :return: a copy of [imgs] with `"processed_path"` pointing to the newly processed images
        """

        img_paths = imgs.keys()
        eyes = {it: imgs[it]["eyes"] for it in img_paths}

        # Find scale for resizing
        eye_dists = {it: math.dist(eyes[it][0], eyes[it][1]) for it in img_paths}
        min_eye_dist = np.min(np.array(list(eye_dists.values())))
        scales = {it: min_eye_dist / eye_dists[it] for it in img_paths}
        scaled_img_dims = {it: (scales[it] * imgs[it]["dims"]).astype(int) for it in img_paths}

        # Find translation to align eyes
        eye_centers = {it: np.mean([eyes[it][0], eyes[it][1]], axis=0).astype(int) for it in img_paths}
        scaled_eye_centers = {it: (scales[it] * eye_centers[it]).astype(int) for it in img_paths}
        max_scaled_eye_center = np.max(np.array(list(scaled_eye_centers.values())), axis=0)
        translations = {it: max_scaled_eye_center - scaled_eye_centers[it] for it in img_paths}

        # Find rotation angle
        # Note that angle is negated because y-axis is flipped by OpenCV, so positive angle is clockwise rotation
        scaled_relative_right_eye_positions = \
            {it: scales[it] * eyes[it][1] - scaled_eye_centers[it] for it in img_paths}
        angles = {k: -math.atan2(v[1], v[0]) for k, v in scaled_relative_right_eye_positions.items()}

        # Find cropping boundaries
        img_corners_after_rotation = {it: rotate(max_scaled_eye_center,
                                                 translations[it] + get_corners(scaled_img_dims[it]),
                                                 angles[it]) for it in img_paths}
        img_inner_boundaries = {it: largest_inner_rectangle(img_corners_after_rotation[it]) for it in img_paths}
        min_inner_boundaries = rectangle_overlap(np.array(list(img_inner_boundaries.values())))
        min_inner_boundaries = (np.floor(min_inner_boundaries / 2) * 2).astype(int)

        # Perform normalization
        processed_imgs = copy.deepcopy(imgs)
        pbar = tqdm(imgs.items(), desc="Normalizing images", file=sys.stdout)
        for img_path, img_data in pbar:
            eyes_string = np.array2string(eyes[img_path])
            params_string = np.array2string(np.hstack([scales[img_path],
                                                       translations[img_path],
                                                       angles[img_path],
                                                       min_inner_boundaries]))

            # Skip if cached
            state_hash = Hasher.hash_string(f"{eyes_string}-{params_string}")
            if self.normalized_cache.has(img_data["hash"], state_hash):
                processed_imgs[img_path]["processed_path"] = self.normalized_cache.path(img_data["hash"], state_hash)
                continue

            # Validate normalization parameters
            angle_abs = math.fabs(math.degrees(angles[img_path]))
            if angle_abs >= 45.0:
                raise UserException(f"Image '{img_path}' is rotated by {angle_abs} degrees, but Facemation only "
                                    f"supports angles up to 45 degrees (but preferably much lower). "
                                    f"You should manually rotate the image and crop out the relevant parts, or remove "
                                    f"the image from the inputs altogether.")
            if angle_abs >= 30.0:
                warnings.warn(f"Image '{img_path}' is rotated by {angle_abs} degrees, which may cause a very small "
                              f"output video. "
                              f"Consider manually cropping out the relevant parts of the image, or removing the image "
                              f"from the inputs altogether.")

            # Normalize image
            translated_dims = tuple(scaled_img_dims[img_path] + translations[img_path])
            translation_matrix = (1, 0, -translations[img_path][0], 0, 1, -translations[img_path][1])

            img = load_image(img_data["processed_path"])
            img = img.resize(scaled_img_dims[img_path])
            img = img.transform(translated_dims, Image.AFFINE, translation_matrix)
            img = img.rotate(-math.degrees(angles[img_path]), center=tuple(max_scaled_eye_center))
            img = img.crop((min_inner_boundaries[0], min_inner_boundaries[1],
                            min_inner_boundaries[2], min_inner_boundaries[3]))

            # Store normalized image
            processed_imgs[img_path]["processed_path"] = self.normalized_cache.cache(img, img_data["hash"], state_hash)

        return processed_imgs


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
