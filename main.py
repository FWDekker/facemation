import glob
import math
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import dlib
import imutils
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from Cache import ImageCache, NdarrayCache
from ConfigHelper import load_config
from HashHelper import sha256sum, sha256sums
from ImageHelper import write_on_image
from UserException import UserException


def find_faces(input_dir: str,
               cache_dir: str,
               error_dir: str,
               face_selection_override: Dict[str, Callable[[dlib.full_object_detection], int]],
               shape_predictor: dlib.shape_predictor) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Finds one face in each image in [input_dir], with each face expressed as the positions of the eyes.

    In addition to returning the eye coordinates, this function also caches results in [cache_dir], which significantly
    speeds up the process.

    Raises a [UserException] if no or multiple faces are found in an image. Additionally, if multiple faces are found,
    the image is written to [error_dir] with debugging information.

    :param input_dir: the directory with original input images
    :param cache_dir: the directory to cache found faces in
    :param error_dir: the directory to write debugging information in to assist the user
    :param shape_predictor: a function that extracts faces from an image
    :param face_selection_override: selects the index of the face to return if multiple faces are found
    :return: a mapping from filenames to left eye positions, and a mapping from filenames to right eye positions
    """

    eyes_left = {}
    eyes_right = {}

    cache = NdarrayCache(cache_dir, "face", ".cache")
    detector = dlib.get_frontal_face_detector()

    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Detecting faces", file=sys.stdout)
    for idx, image_path in enumerate(pbar):
        image_name = os.path.basename(image_path)
        image_hash = sha256sum(image_path)

        # Read from cache if exists
        if cache.has(image_hash, []):
            eyes_both = cache.load(image_hash, [])
            eyes_left[image_name] = eyes_both[0]
            eyes_right[image_name] = eyes_both[1]
            continue

        # Detect eyes
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detector(image_rgb, 1)

        faces = dlib.full_object_detections()
        for detection in detections:
            faces.append(shape_predictor(image_rgb, detection))

        # Determine what to do if there are multiple faces
        if len(faces) == 0:
            raise UserException(f"Not enough faces: Found 0 faces in '{image_path}'.")
        elif len(faces) > 1:
            if image_name in face_selection_override:
                face = sorted(list(faces), key=face_selection_override[image_name])[0]
            else:
                bb = [it.rect for it in faces]
                bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
                for it in bb:
                    image = cv2.rectangle(image, it[0], it[1], (255, 0, 0), 5)
                cv2.imwrite(f"{error_dir}/{image_name}", image)

                raise UserException(f"Too many faces: Found {len(faces)} in '{image_path}'. "
                                    f"The image has been stored in '{Path(error_dir).absolute()}' with squares drawn "
                                    f"around all faces that were found. "
                                    f"You can select which face should be used by adjusting the "
                                    f"'face_selection_override' option; "
                                    f"see 'config_default.py' for more information.")
        else:
            face = faces[0]

        # Store results
        eyes_left[image_name] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(42, 48)]), axis=0)
        eyes_right[image_name] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(36, 42)]), axis=0)
        cache.cache(image_hash, [], np.vstack([eyes_left[image_name], eyes_right[image_name]]))

    return eyes_left, eyes_right


def normalize_images(input_dir: str,
                     cache_dir: str,
                     draw_debug: bool,
                     eyes_left: Dict[str, np.ndarray],
                     eyes_right: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """
    Translates, rotates, and resizes each file in [input_dir], storing the results in [cache_dir].

    :param input_dir: the directory with original input images
    :param cache_dir: the directory to cache normalized images in
    :param draw_debug: `True` if eyes should be drawn in all images in [cache_dir]
    :param eyes_left: a mapping from filenames in [input_dir] to left eye positions
    :param eyes_right: a mapping from filenames in [input_dir] to right eye positions
    :return: the smallest width and smallest height observed in all images
    """

    width_min, height_min = [1e99, 1e99]

    cache = ImageCache(cache_dir, "normalized", ".jpg")

    # Determine eye position targets
    eyes_dist = {it: math.dist(eyes_left[it], eyes_right[it]) for it in eyes_left.keys()}
    eyes_dist_avg = np.mean(np.array(list(eyes_dist.values())))

    eyes_left_np = np.array(list(eyes_left.values()))
    eyes_right_np = np.array(list(eyes_right.values()))

    eyes_left_avg = np.mean(eyes_left_np, axis=0)
    eyes_right_avg = np.mean(eyes_right_np, axis=0)
    eyes_center_avg = np.mean([eyes_left_avg, eyes_right_avg], axis=0)

    # Translate, rotate, resize
    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Translating, rotating, resizing", file=sys.stdout)
    for idx, image_path in enumerate(pbar):
        image_name = os.path.basename(image_path)
        image_hash = sha256sum(image_path)

        eye_left, eye_right = eyes_left[image_name], eyes_right[image_name]
        eye_both = np.vstack([eyes_left[image_name], eyes_right[image_name]])
        eye_hash = sha256sums(np.array2string(eye_both))

        # Read from cache if exists
        if cache.has(image_hash, [eye_hash]):
            image = cache.load(image_hash, [eye_hash])
        else:
            # Read image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            eye_center = np.mean([eye_left, eye_right], axis=0)

            # Draw debug information
            if draw_debug:
                image = cv2.circle(image, eye_left.astype(int), radius=20, color=(0, 255, 0), thickness=-1)
                image = cv2.circle(image, eye_right.astype(int), radius=20, color=(0, 255, 0), thickness=-1)
                image = cv2.circle(image, eyes_left_avg.astype(int), radius=20, color=(0, 0, 255), thickness=-1)
                image = cv2.circle(image, eyes_right_avg.astype(int), radius=20, color=(0, 0, 255), thickness=-1)
                image = cv2.circle(image, eye_center.astype(int), radius=20, color=(255, 0, 0), thickness=-1)

            # Translate
            eyes_center = np.mean([eye_left, eye_right], axis=0)
            translation = [eyes_center_avg[0] - eyes_center[0], eyes_center_avg[1] - eyes_center[1]]
            transformation = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            image = cv2.warpAffine(image, transformation, (width, height))

            # Rotate
            eye_right_relative = eye_right - eyes_center
            angle = math.atan(eye_right_relative[1] / eye_right_relative[0])
            image = imutils.rotate(image, angle)

            # Resize
            scale = eyes_dist_avg / eyes_dist[image_name]
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

            # Store normalized image
            cache.cache(image_hash, [eye_hash], image)

        # Store smallest image seen
        height_new, width_new = image.shape[:2]
        width_min, height_min = min(width_min, width_new), min(height_min, height_new)

    # Make dimensions even
    width_min = width_min if width_min % 2 == 0 else width_min - 1
    height_min = height_min if height_min % 2 == 0 else height_min - 1

    return width_min, height_min


def crop_images(input_dir: str, cache_dir: str, cropped_width: int, cropped_height: int) -> None:
    """
    Crops each image in [input_dir] to be [width] by [height] pixels.

    :param input_dir: the directory with original input images
    :param cache_dir: the directory to read normalized images from and cache cropped images in
    :param cropped_width: the width that each image should be
    :param cropped_height: the height that each image should be
    :return: `None`
    """

    normalized_cache = ImageCache(cache_dir, "normalized", ".jpg")
    cropped_cache = ImageCache(cache_dir, "cropped", ".jpg")

    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Cropping", file=sys.stdout)
    for idx, image_path in enumerate(pbar):
        image_hash = sha256sum(image_path)
        cropped_size_hash = sha256sums(np.array2string(np.array([cropped_width, cropped_height])))

        if cropped_cache.has(image_hash, [cropped_size_hash]):
            pass
        else:
            image = normalized_cache.load_any(image_hash)
            height, width = image.shape[:2]

            width_excess = width - cropped_width
            width_start = int(width_excess / 2)
            width_end = width_start + cropped_width

            height_excess = height - cropped_height
            height_start = int(height_excess / 2)
            height_end = height_start + cropped_height

            image = image[height_start:height_end, width_start:width_end]
            cropped_cache.cache(image_hash, [cropped_size_hash], image)


def add_captions(input_dir: str, cache_dir: str, filename_to_date: Callable[[str], date],
                 date_to_caption: Callable[[date], str]) -> None:
    """
    Adds a caption to each image in [input_dir] based on [filename_to_date] and [date_to_caption], storing the captioned
    images in [cache_dir], renaming them based on their natsort indices.

    Raises a [UserException] if [date_to_caption] raises an exception.

    :param input_dir: the directory with original input images
    :param cache_dir: the directory to read cropped images from and cache captioned images in
    :param filename_to_date: converts a filename to a [date]
    :param date_to_caption: converts a [date] to a caption
    :return: `None`
    """

    cropped_cache = ImageCache(cache_dir, "cropped", ".jpg")
    captioned_cache = ImageCache(cache_dir, "captioned", ".jpg")

    pbar = tqdm(natsorted(glob.glob(f"{input_dir}/*.jpg")), desc="Adding captions", file=sys.stdout)
    for idx, image_path in enumerate(pbar):
        image_name = os.path.basename(image_path)

        try:
            caption = date_to_caption(filename_to_date(image_name))
        except Exception as exception:
            pbar.close()
            raise UserException("Failed to convert date to caption. "
                                "Your 'filename_to_date' has been configured wrongly. "
                                "Check your configuration for more details.", exception) from None

        image_hash = sha256sum(image_path)
        caption_hash = sha256sums(caption)
        if captioned_cache.has(image_hash, [caption_hash]):
            continue

        image = cropped_cache.load_any(image_hash)
        image = write_on_image(image, caption, (0.05, 0.95), 0.05)
        captioned_cache.cache(image_hash, [caption_hash], image)


def demux_images(enabled: bool, input_dir: str, cache_dir: str, frames_dir: str, output_path: str, fps: int, crf: int,
                 codec: str, video_filters: list[str]) -> None:
    """
    Demuxes the images in [input_dir] into a video in [output_path] using FFmpeg.

    Raises a [UserException] if FFmpeg has a non-zero exit code.

    :param enabled: `True` if and only if this function should run
    :param input_dir: the directory with original input images
    :param cache_dir: the directory to read processed images from
    :param frames_dir: the directory to store frames in for FFmpeg
    :param output_path: the path relative to [input_dir] to save the created video as
    :param fps: the frames per second
    :param crf: the constant rate factor
    :param codec: the codec to encode the video with
    :param video_filters: the filters to apply to the video stream
    :return: `None`
    """

    if enabled:
        # TODO: Symlink to captioned files!
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
            ], cwd=input_dir, stderr=sys.stdout, check=True)
        except Exception as exception:
            raise UserException("FFmpeg failed to create a video. "
                                "Read the messages above for more information.", exception) from None


def main() -> None:
    """
    Main entry point.

    :return: `None`
    """
    cfg = load_config()

    # Clean up from previous runs
    if Path(cfg.error_dir).exists():
        shutil.rmtree(cfg.error_dir)
    if Path(cfg.frames_dir).exists():
        shutil.rmtree(cfg.frames_dir)
    Path(cfg.output_path).unlink(missing_ok=True)

    Path(cfg.input_dir).mkdir(exist_ok=True)
    Path(cfg.error_dir).mkdir(exist_ok=True)
    Path(cfg.cache_dir).mkdir(exist_ok=True)
    Path(cfg.frames_dir).mkdir(exist_ok=True)

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
    shape_predictor = dlib.shape_predictor(cfg.shape_predictor)

    if (not Path(cfg.input_dir).exists()) or len(glob.glob(f"{cfg.input_dir}/*.jpg")) == 0:
        print(f"No images detected in '{Path(cfg.input_dir).absolute()}'. "
              f"Are you sure you put them in the right place?",
              file=sys.stderr)
        return

    # Run facemation
    try:
        # Pre-process
        eyes_left, eyes_right = find_faces(cfg.input_dir, cfg.cache_dir, cfg.error_dir, cfg.face_selection_override,
                                           shape_predictor)

        # Process
        width_min, height_min = normalize_images(cfg.input_dir, cfg.cache_dir, cfg.normalize_draw_debug,
                                                 eyes_left, eyes_right)
        crop_images(cfg.input_dir, cfg.cache_dir, width_min, height_min)

        # Post-process
        add_captions(cfg.input_dir, cfg.cache_dir, cfg.filename_to_date, cfg.date_to_caption)

        # Demux
        demux_images(cfg.ffmpeg_enabled, cfg.input_dir, cfg.cache_dir, cfg.frames_dir, cfg.output_path,
                     cfg.ffmpeg_fps, cfg.ffmpeg_crf, cfg.ffmpeg_codec, cfg.ffmpeg_video_filters)

        print("Done!")
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)


if __name__ == "__main__":
    main()
