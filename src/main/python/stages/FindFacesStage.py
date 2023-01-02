import functools
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable

import cv2
import dlib
import numpy as np
from tqdm.contrib.concurrent import process_map

import Files
import Resolver
from Cache import NdarrayCache
from Pipeline import PreprocessingStage, ImageInfo
from UserException import UserException

"""A face expressed as the (x, y)-coordinates of the eyes, with the left-most eye in the picture as the first row."""
Face = np.ndarray
"""Sorting function for faces to determine which face to select if multiple faces are found."""
FaceSelectionOverride = Callable[[dlib.full_object_detection], int]

# Global field because this cannot be pickled between parallel processes
g_face_selection_overrides: Dict[str, FaceSelectionOverride]


class FindFacesStage(PreprocessingStage):
    """
    Finds faces in the input images.
    """

    error_dir: str
    face_cache: NdarrayCache
    face_selection_overrides: Dict[str, FaceSelectionOverride]

    def __init__(self, cache_dir: str, error_dir: str, face_selection_overrides: Dict[str, FaceSelectionOverride]):
        """
        Constructs a new `FindFacesStage`.

        :param cache_dir: the directory to cache found faces in
        :param error_dir: the directory to write debugging information in to assist the user
        :param face_selection_overrides: if multiple faces are found in an image, the file path is used as a key to find
        the `Callable` by which the faces are sorted, and the first face is used
        """

        Files.cleardir(error_dir)

        self.error_dir = error_dir
        self.face_cache = NdarrayCache(cache_dir, "face", ".cache")
        self.face_selection_overrides = face_selection_overrides

    def preprocess(self, imgs: Dict[Path, ImageInfo]) -> Dict[Path, ImageInfo]:
        """
        Finds one face in each image in [imgs], with each face expressed as the positions of the eyes, additionally
        caching the face data in [self.face_cache].

        Raises a [UserException] if no or multiple faces are found in an image, and override is configured in
        [self.face_selection_overrides]. Additionally, if an exception is thrown, the image is written to
        [self.error_dir] with debugging information.

        :param imgs: a read-only mapping from original input paths to the preprocessed data obtained thus far
        :return: a mapping from original input path to the face in the image
        """

        global g_face_selection_overrides
        g_face_selection_overrides = self.face_selection_overrides

        return dict(process_map(functools.partial(find_face, face_cache=self.face_cache, error_dir=self.error_dir),
                                imgs.items(),
                                desc="Detecting faces",
                                file=sys.stdout))


def find_face(img: Tuple[Path, ImageInfo], face_cache: NdarrayCache, error_dir: str) -> Tuple[Path, ImageInfo]:
    """
    Finds the face in [img], expressed as the positions of the eyes, caching the face data in [face_cache].

    Raises a [UserException] if no or multiple faces are found in an image, and [g_face_selection_overrides] is not
    configured for this image. Additionally, if an exception is thrown, the image is written to [error_dir] with
    visualized debugging information.

    :param img: the original input path and the pre-processing data of the image to find a face in
    :param face_cache: the cache to store the found face in
    :param error_dir: the directory to write debugging information in to assist the user
    :return: the original input path and the found face
    """

    img_path, img_data = img
    if face_cache.has(img_data["hash"]):
        return img_path, {"eyes": face_cache.load(img_data["hash"])}

    # Initialize face recognition
    global g_face_selection_overrides
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(str(Resolver.resource_path("shape_predictor_68_face_landmarks.dat")))

    # Find face
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = dlib.full_object_detections()
    detections = detector(img_rgb, 1)
    for detection in detections:
        faces.append(shape_predictor(img_rgb, detection))

    # Determine what to do if there are multiple faces
    if len(faces) == 0:
        raise UserException(f"Not enough faces: Found 0 faces in '{img_path}'.")
    elif len(faces) == 1:
        face = faces[0]
    else:
        img_name = img_path.name  # Includes file extension

        if img_name in g_face_selection_overrides:
            face = sorted(list(faces), key=g_face_selection_overrides[img_name])[0]
        else:
            bb = [it.rect for it in faces]
            bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
            for it in bb:
                img = cv2.rectangle(img, it[0], it[1], (255, 0, 0), 5)
            cv2.imwrite(f"{error_dir}/{img_name}", img)

            raise UserException(f"Too many faces: Found {len(faces)} in '{img_path}'. "
                                f"The image has been stored in '{error_dir}' with squares drawn around all faces that "
                                f"were found. "
                                f"You can select which face should be used by adjusting the 'face_selection_override' "
                                f"option; "
                                f"see 'config_default.py' for more information.")

    # Store results
    eyes = np.vstack([(np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(36, 42)]), axis=0)),
                      (np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(42, 48)]), axis=0))])
    face_cache.cache(eyes, img_data["hash"])

    return img_path, {"eyes": eyes}
