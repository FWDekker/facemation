import math

import numpy as np


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
