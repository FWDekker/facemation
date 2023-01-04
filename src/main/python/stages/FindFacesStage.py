import functools
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, TypedDict

import dlib
import numpy as np
from PIL import ImageDraw
from tqdm.contrib.concurrent import process_map

import Files
import Resolver
from Cache import NdarrayCache
from ImageLoader import load_image
from Pipeline import PreprocessingStage, ImageInfo
from UserException import UserException

FaceSelectionOverride = Callable[[dlib.full_object_detection], int]
FindFacesConfig = TypedDict("FindFacesConfig", {"error_dir": str,
                                                "face_selection_overrides": Dict[str, FaceSelectionOverride]})
Face = np.ndarray  # (x, y)-coordinates of the eyes, with the left-most eye in the picture as the first row

# Global field because this cannot be pickled between parallel processes
g_face_selection_overrides: Dict[str, FaceSelectionOverride]


class FindFacesStage(PreprocessingStage):
    """
    Finds faces in the input images.
    """

    cfg: FindFacesConfig
    face_cache: NdarrayCache

    def __init__(self, cfg: FindFacesConfig, cache_dir: str):
        """
        Constructs a new `FindFacesStage`.

        :param cfg: the configuration for this stage
        :param cache_dir: the directory to cache found faces in
        """

        Files.cleardir(cfg["error_dir"])

        self.cfg = cfg
        self.face_cache = NdarrayCache(cache_dir, "face", ".cache")

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
        g_face_selection_overrides = self.cfg["face_selection_overrides"]

        return dict(process_map(functools.partial(find_face,
                                                  face_cache=self.face_cache,
                                                  error_dir=self.cfg["error_dir"]),
                                imgs.items(),
                                desc="Detecting faces",
                                file=sys.stdout))


def find_face(img_tuple: Tuple[Path, ImageInfo], face_cache: NdarrayCache, error_dir: str) -> Tuple[Path, ImageInfo]:
    """
    Finds the face in [img_tuple], expressed as the positions of the eyes, caching the face data in [face_cache].

    Raises a [UserException] if no or multiple faces are found in an image, and [g_face_selection_overrides] is not
    configured for this image. Additionally, if an exception is thrown, the image is written to [error_dir] with
    visualized debugging information.

    :param img_tuple: the original input path and the pre-processing data of the image to find a face in
    :param face_cache: the cache to store the found face in
    :param error_dir: the directory to write debugging information in to assist the user
    :return: the original input path and the found face
    """

    img_path, img_data = img_tuple
    if face_cache.has(img_data["hash"]):
        return img_path, {"eyes": face_cache.load(img_data["hash"])}

    # Initialize face recognition
    global g_face_selection_overrides
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(str(Resolver.resource_path("shape_predictor_5_face_landmarks.dat")))

    # Find face
    img = load_image(img_path)
    img_np = np.array(img)
    faces = dlib.full_object_detections()
    detections = detector(img_np, 1)
    for detection in detections:
        faces.append(shape_predictor(img_np, detection))

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
            img_draw = ImageDraw.Draw(img)

            bb = [it.rect for it in faces]
            bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
            for it in bb:
                img_draw.rectangle(it[0], it[1], (255, 0, 0), 5)

            img.save(f"{error_dir}/{img_name}")

            raise UserException(f"Too many faces: Found {len(faces)} in '{img_path}'. "
                                f"The image has been stored in '{error_dir}' with squares drawn around all faces that "
                                f"were found. "
                                f"You can select which face should be used by adjusting the 'face_selection_override' "
                                f"option; "
                                f"see 'config_default.py' for more information.")

    # Store results
    eyes = np.vstack([(np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(2, 4)]), axis=0)),
                      (np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(0, 2)]), axis=0))])
    face_cache.cache(eyes, img_data["hash"])

    return img_path, {"eyes": eyes}
