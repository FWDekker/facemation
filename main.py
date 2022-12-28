import functools
import glob
import math
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict, TypedDict, Tuple

import cv2
import dlib
import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from Cache import ImageCache, NdarrayCache
from ConfigHelper import load_config
from HashHelper import sha256sum, sha256sums
from ImageHelper import write_on_image
from MathHelper import rotate, get_corners, largest_inner_rectangle, rectangle_overlap
from UserException import UserException

Coords = np.ndarray  # x, y
Dimensions = np.ndarray  # width, height
MetaData = TypedDict("MetaData", {"hash": str, "dims": Dimensions})


def read_image_data(input_dir: str) -> Dict[str, MetaData]:
    """
    Reads image meta-data, such as filesize and image contents hash.

    :param input_dir: the directory to read input files from
    :return: a mapping from input images to the hash of the image and the dimensions of the image
    """

    image_data = {}

    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Reading image meta-data", file=sys.stdout)
    for image_path in pbar:
        image_hash = sha256sum(image_path)

        image = Image.open(image_path)
        width, height = image.size
        exif = image.getexif().get(0x0112)
        if exif == 6 or exif == 8:
            width, height = height, width

        image_data[image_path] = {"hash": image_hash, "dims": np.array([width, height])}

    return image_data


def find_face(img: Tuple[str, MetaData], face_cache: NdarrayCache, error_dir: str) -> None:
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

    # Find face
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector(img_rgb, 1)

    faces = dlib.full_object_detections()
    for detection in detections:
        faces.append(shape_predictor(img_rgb, detection))

    # Determine what to do if there are multiple faces
    if len(faces) == 0:
        raise UserException(f"Not enough faces: Found 0 faces in '{img_path}'.")
    elif len(faces) > 1:
        img_name = os.path.basename(img_path)  # Includes file extension

        if img_name in cfg.face_selection_override:
            face = sorted(list(faces), key=cfg.face_selection_override[img_name])[0]
        else:
            bb = [it.rect for it in faces]
            bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
            for it in bb:
                img = cv2.rectangle(img, it[0], it[1], (255, 0, 0), 5)
            cv2.imwrite(f"{error_dir}/{img_name}", img)

            raise UserException(f"Too many faces: Found {len(faces)} in '{img_path}'. "
                                f"The image has been stored in '{Path(error_dir).absolute()}' with squares drawn "
                                f"around all faces that were found. "
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


def find_all_faces(imgs: Dict[str, MetaData], face_cache: NdarrayCache, error_dir: str) -> None:
    """
    Finds one face in each image in [imgs], with each face expressed as the positions of the eyes, caching the face data
    in [face_cache].

    Raises a [UserException] if no or multiple faces are found in an image. Additionally, if multiple faces are found,
    the image is written to [error_dir] with debugging information.

    :param imgs: the metadata of the images to detect faces in
    :param face_cache: the cache to store found faces in
    :param error_dir: the directory to write debugging information in to assist the user
    :return: `None`
    """

    process_map(functools.partial(find_face, face_cache=face_cache, error_dir=error_dir), imgs.items(),
                desc="Detecting faces",
                file=sys.stdout)


def normalize_images(imgs: Dict[str, MetaData],
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
    img_corners_after_rotation = {it: rotate(max_scaled_eye_center,
                                             translations[it] + get_corners(scaled_img_dims[it]),
                                             angles[it]) for it in img_paths}
    img_inner_boundaries = {it: largest_inner_rectangle(img_corners_after_rotation[it]) for it in img_paths}
    min_inner_boundaries = rectangle_overlap(np.array(list(img_inner_boundaries.values())))
    min_inner_boundaries = (np.floor(min_inner_boundaries / 2) * 2).astype(int)

    # Perform normalization
    pbar = tqdm(imgs.items(), desc="Normalizing images", file=sys.stdout)
    for img_path, img_data in pbar:
        eye_hash = sha256sums(np.array2string(eyes[img_path]))
        normalization_hash = sha256sums(np.array2string(np.hstack([scales[img_path],
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


def add_captions(imgs: Dict[str, MetaData],
                 input_cache: ImageCache,
                 captioned_cache: ImageCache,
                 filename_to_date: Callable[[str], date],
                 date_to_caption: Callable[[date], str]) -> None:
    """
    For each image in [imgs], finds the corresponding image in [input_cache], and adds a caption using
    [filename_to_date] and [date_to_caption], storing the captioned images in [captioned_cache].

    Raises a [UserException] if [date_to_caption] raises an exception.

    :param imgs: the metadata of the images from which the normalized inputs are derived
    :param input_cache: the cache to read the images to caption from, with keys matching those in [imgs]
    :param captioned_cache: the cache to store captioned images in
    :param filename_to_date: converts a filename to a [date]
    :param date_to_caption: converts a [date] to a caption
    :return: `None`
    """

    pbar = tqdm(imgs.items(), desc="Adding captions", file=sys.stdout)
    for img_path, img_data in pbar:
        img_name = os.path.basename(img_path)

        try:
            # TODO: Define `caption` as single function with access to all image metadata
            caption = date_to_caption(filename_to_date(img_name))
        except Exception as exception:
            pbar.close()
            raise UserException(f"Failed to convert date to caption for image '{img_name}'. "
                                f"Your 'filename_to_date' has been configured wrongly. "
                                f"Check your configuration for more details.", exception) from None

        # TODO: Base cache key on hash of previous image
        caption_hash = sha256sums(caption)
        if captioned_cache.has(img_data["hash"], [caption_hash]):
            continue

        img = input_cache.load_any(img_data["hash"])
        img = write_on_image(img, caption, (0.05, 0.95), 0.05)
        captioned_cache.cache(img_data["hash"], [caption_hash], img)


def demux_images(enabled: bool,
                 imgs: Dict[str, MetaData],
                 input_cache: ImageCache,
                 frames_dir: str,
                 output_path: str,
                 fps: int,
                 crf: int,
                 codec: str,
                 video_filters: list[str]) -> None:
    """
    Given the original input image in [imgs], selects the corresponding processed images from [input_cache] and stores
    these in [frames_dir], and demuxes the contents of [frames_dir] into video in [output_path] using FFmpeg.

    Raises a [UserException] if FFmpeg has a non-zero exit code.

    :param enabled: `True` if and only if this function should run
    :param imgs: the metadata of the images from which the inputs are derived
    :param input_cache: the cache to select frames to process from
    :param frames_dir: the directory to store frame links in for FFmpeg
    :param output_path: the path relative to [input_dir] to save the created video as
    :param fps: the frames per second
    :param crf: the constant rate factor
    :param codec: the codec to encode the video with
    :param video_filters: the filters to apply to the video stream
    :return: `None`
    """

    if enabled:
        pbar = tqdm(natsorted(imgs.keys()), desc="Selecting frames", file=sys.stdout)
        for idx, image_path in enumerate(pbar):
            captioned_path = input_cache.get_path_any(imgs[image_path]["hash"])
            os.symlink(os.path.relpath(captioned_path, frames_dir), f"{frames_dir}/{idx}.jpg")

        print("Demuxing into video:")
        try:
            subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-stats",
                "-y",
                "-f", "image2",
                "-r", fps,
                "-i", "%d.jpg",
                "-vcodec", codec,
                "-crf", crf,
                "-vf", ",".join(video_filters),
                output_path
            ], cwd=frames_dir, stderr=sys.stdout, check=True)
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None


def main() -> None:
    """
    Main entry point.

    :return: `None`
    """

    # Clean up from previous runs
    if Path(cfg.error_dir).exists():
        shutil.rmtree(cfg.error_dir)
    if Path(cfg.frames_dir).exists():
        shutil.rmtree(cfg.frames_dir)
    Path(cfg.output_path).unlink(missing_ok=True)

    Path(cfg.input_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.error_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.frames_dir).mkdir(parents=True, exist_ok=True)

    # Validate requirements and inputs
    if cfg.ffmpeg_enabled and shutil.which("ffmpeg") is None:
        print(f"FFmpeg is enabled in your configuration but is not installed. "
              f"Check the README for more information on the requirements.", file=sys.stderr)
        return

    if not Path(cfg.shape_predictor).exists():
        print(f"Face detector '{Path(cfg.shape_predictor).absolute()}' could not be found. "
              f"Make sure to download the file from the link in the README and place it in the same directory as "
              f"'main.py'.", file=sys.stderr)
        return

    if (not Path(cfg.input_dir).exists()) or len(glob.glob(f"{cfg.input_dir}/*.jpg")) == 0:
        print(f"No images detected in '{Path(cfg.input_dir).absolute()}'. "
              f"Are you sure you put them in the right place?",
              file=sys.stderr)
        return

    # Run facemation
    try:
        face_cache = NdarrayCache(cfg.cache_dir, "face", ".cache")
        normalized_cache = ImageCache(cfg.cache_dir, "normalized", ".jpg")
        captioned_cache = ImageCache(cfg.cache_dir, "captioned", ".jpg")

        imgs = read_image_data(cfg.input_dir)
        find_all_faces(imgs, face_cache, cfg.error_dir)
        normalize_images(imgs, face_cache, normalized_cache)
        add_captions(imgs, normalized_cache, captioned_cache, cfg.filename_to_date, cfg.date_to_caption)
        demux_images(cfg.ffmpeg_enabled, imgs, captioned_cache, cfg.frames_dir, cfg.output_path, cfg.ffmpeg_fps,
                     cfg.ffmpeg_crf, cfg.ffmpeg_codec, cfg.ffmpeg_video_filters)

        print("Done!")
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)


if __name__ == "__main__":
    # Create globals to reduce process communication
    cfg = load_config()
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(cfg.shape_predictor)

    # Invoke main
    main()
