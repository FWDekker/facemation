import functools
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable

import cv2
import dlib
import numpy as np
from tqdm.contrib.concurrent import process_map

import Resolver
from Cache import NdarrayCache
from ReadInputsStage import ImageMetadata
from UserException import UserException

# Global field because this cannot be pickled between processes
g_face_selection_override: Dict[str, Callable[[dlib.full_object_detection], int]]


def find_all_faces(imgs: Dict[str, ImageMetadata], face_cache: NdarrayCache,
                   face_selection_override: Dict[str, Callable[[dlib.full_object_detection], int]],
                   error_dir: str) -> None:
    """
    Finds one face in each image in [imgs], with each face expressed as the positions of the eyes, caching the face data
     in [face_cache].

    Raises a [UserException] if no or multiple faces are found in an image. Additionally, if multiple faces are found,
    the image is written to [error_dir] with debugging information.

    :param imgs: the metadata of the images to detect faces in
    :param face_cache: the cache to store found faces in
    :param face_selection_override: if multiple faces are found in an image, the filename is used as a key to find the
    callable by which the faces are sorted, and the first face is used
    :param error_dir: the directory to write debugging information in to assist the user
    :return: `None`
    """

    global g_face_selection_override
    g_face_selection_override = face_selection_override

    process_map(functools.partial(_find_face, face_cache=face_cache, error_dir=error_dir), imgs.items(),
                desc="Detecting faces",
                file=sys.stdout)


def _find_face(img: Tuple[str, ImageMetadata], face_cache: NdarrayCache, error_dir: str) -> None:
    """
    Finds the face in [img], expressed as the positions of the eyes, caching the face data in [face_cache].

    Raises a [UserException] if no or multiple faces are found in an image, and [cfg.face_selection_override] is not
    configured for this image. Additionally, if an exception is thrown, the image is written to [error_dir] with
    visualized debugging information.

    :param img: the path to and metadata of the image to find the face in
    :param face_cache: the cache to store the found face in
    :param error_dir: the directory to write debugging information in to assist the user
    :return: `None`
    """

    img_path, img_data = img
    if face_cache.has(img_data["hash"], []):
        return

    # Initialize face recognition
    global g_face_selection_override
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(str(Resolver.resource_path("shape_predictor_68_face_landmarks.dat")))

    # Find face
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = dlib.full_object_detections()
    detections = detector(img_rgb, 1)
    for detection in detections:
        faces.append(shape_predictor(img_rgb, detection))

    # Determine what to do if there are multiple faces
    if len(faces) == 0:
        raise UserException(f"Not enough faces: Found 0 faces in '{img_path}'.")
    elif len(faces) > 1:
        img_name = os.path.basename(img_path)  # Includes file extension

        if img_name in g_face_selection_override:
            face = sorted(list(faces), key=g_face_selection_override[img_name])[0]
        else:
            bb = [it.rect for it in faces]
            bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
            for it in bb:
                img = cv2.rectangle(img, it[0], it[1], (255, 0, 0), 5)
            cv2.imwrite(f"{error_dir}/{img_name}", img)

            raise UserException(f"Too many faces: Found {len(faces)} in '{img_path}'. "
                                f"The image has been stored in '{Path(error_dir).resolve()}' with squares drawn around "
                                f"all faces that were found. "
                                f"You can select which face should be used by adjusting the 'face_selection_override' "
                                f"option; "
                                f"see 'config_default.py' for more information.")
    else:
        face = faces[0]

    # Store results
    # Note that the "left eye" is the left-most eye in the image, i.e. the anatomical "right eye"
    left_eye = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(36, 42)]), axis=0)
    right_eye = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(42, 48)]), axis=0)
    face_cache.cache(img_data["hash"], [], np.vstack([left_eye, right_eye]))
