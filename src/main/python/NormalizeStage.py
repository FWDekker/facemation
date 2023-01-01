import math
import sys
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

import SHA256
from Cache import NdarrayCache, ImageCache
from ReadInputsStage import ImageMetadata
from UserException import UserException


def normalize_images(imgs: Dict[str, ImageMetadata],
                     face_cache: NdarrayCache,
                     normalized_cache: ImageCache) -> None:
    """
    Translates, rotates, and resizes each file in [imgs], storing the results in [normalized_cache].

    :param imgs: the metadata of the images to normalize
    :param face_cache: the cache to read found faces from
    :param normalized_cache: the cache to store normalized images in
    :return: `None`
    """

    img_paths = imgs.keys()
    eyes = {it: face_cache.load(imgs[it]["hash"], []) for it in img_paths}

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
    # Note that angle is negated because the y-axis is flipped by OpenCV, so a positive angle is a clockwise rotation
    scaled_relative_right_eye_positions = {it: scales[it] * eyes[it][1] - scaled_eye_centers[it] for it in img_paths}
    angles = {k: -math.atan2(v[1], v[0]) for k, v in scaled_relative_right_eye_positions.items()}

    # Find cropping boundaries
    img_corners_after_rotation = {it: _rotate(max_scaled_eye_center,
                                              translations[it] + _get_corners(scaled_img_dims[it]),
                                              angles[it]) for it in img_paths}
    img_inner_boundaries = {it: _largest_inner_rectangle(img_corners_after_rotation[it]) for it in img_paths}
    min_inner_boundaries = _rectangle_overlap(np.array(list(img_inner_boundaries.values())))
    min_inner_boundaries = (np.floor(min_inner_boundaries / 2) * 2).astype(int)

    # Perform normalization
    pbar = tqdm(imgs.items(), desc="Normalizing images", file=sys.stdout)
    for img_path, img_data in pbar:
        eye_hash = SHA256.hash_string(np.array2string(eyes[img_path]))
        normalization_hash = SHA256.hash_string(np.array2string(np.hstack([scales[img_path],
                                                                           translations[img_path],
                                                                           angles[img_path],
                                                                           min_inner_boundaries])))

        # Skip if cached
        if normalized_cache.has(img_data["hash"], [eye_hash, normalization_hash]):
            continue

        # Read image
        img = cv2.imread(img_path)

        # Resize
        img = cv2.resize(img, scaled_img_dims[img_path])

        # Translate
        translation = np.float32([[1, 0, translations[img_path][0]], [0, 1, translations[img_path][1]]])
        img = cv2.warpAffine(img, translation, scaled_img_dims[img_path] + translations[img_path])

        # Rotate
        if math.fabs(math.degrees(angles[img_path])) >= 45.0:
            raise UserException(f"Image '{img_path}' is rotated by {math.degrees(angles[img_path])}, but Facemation "
                                f"only supports angles up to 45 degrees (but preferably much lower)."
                                f"You should manually rotate the image and crop out the relevant parts, or remove the "
                                f"image from the inputs altogether.")

        rotation = cv2.getRotationMatrix2D(max_scaled_eye_center.astype(float), -math.degrees(angles[img_path]), 1.0)
        img = cv2.warpAffine(img, rotation, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Crop
        img = img[min_inner_boundaries[1]:min_inner_boundaries[3], min_inner_boundaries[0]:min_inner_boundaries[2]]

        # Store normalized image
        normalized_cache.cache(img_data["hash"], [eye_hash, normalization_hash], img)


def _get_corners(dims: np.ndarray) -> np.ndarray:
    """
    Returns the corners of a rectangle at `(0, 0)` with width and height as specified in [dims].

    :param dims: the width and height of the rectangle
    :return: the corners of a rectangle at `(0, 0)` with width and height as specified in [dims]
    """

    return np.array([[dims[0], 0], [0, 0], [0, dims[1]], dims])


def _rotate(origin: np.ndarray, points: np.ndarray, angle: float) -> np.ndarray:
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


def _largest_inner_rectangle(corners: np.ndarray) -> np.ndarray:
    """
    Returns the largest non-rotated inner rectangle of the rectangle specified by [corners].

    :param corners: the corners of the rectangle to find the inner rectangle in
    :return: the largest non-rotated inner rectangle of the rectangle specified by [corners]
    """

    xs = np.sort(corners[:, 0])
    ys = np.sort(corners[:, 1])

    return np.array([[xs[1], ys[1]], [xs[2], ys[2]]])


def _rectangle_overlap(rectangles: np.ndarray) -> np.ndarray:
    """
    Returns the largest rectangle that is within all [rectangles], assuming such a rectangle exists.

    :param rectangles: the rectangle to find the overlap of
    :return: the largest rectangle that is within all [rectangles], assuming such a rectangle exists
    """

    return np.array([np.max(rectangles[:, 0, 0]),
                     np.max(rectangles[:, 0, 1]),
                     np.min(rectangles[:, 1, 0]),
                     np.min(rectangles[:, 1, 1])])