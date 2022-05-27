import glob
import hashlib
import math
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from traceback import print_exception
from types import SimpleNamespace

import cv2
import dlib
import imutils
import numpy as np
from natsort import natsorted
from tqdm import tqdm


# Loads config from `config_default.py`, and overrides it with config in `config.py` if it exists.
def load_config() -> SimpleNamespace:
    from config_default import config as cfg_default
    if Path("config.py").exists():
        from config import config as cfg_user
    else:
        cfg_user = {}

    return SimpleNamespace(**(cfg_default | cfg_user))


# Calculates hash of given file
# Taken from https://stackoverflow.com/a/44873382
def sha256sum(filename: str) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        # noinspection PyUnresolvedReferences
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


# Writes [text] on the image at [pos] (being coordinates in the range [0, 1]) with [text_height] as its height relative
# to the image's height.
def write_on_image(image: np.ndarray, text: str, pos: [float, float], text_height: float) -> np.ndarray:
    height, width = image.shape[:2]
    text_scale = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=32)
    text_scale = text_height / (text_scale[0][1] / height)
    text_pos = (math.floor(pos[0] * width), math.floor(pos[1] * height))

    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(0, 0, 0), thickness=32, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, text_pos,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale,
                        color=(255, 255, 255), thickness=16, lineType=cv2.LINE_AA)
    return image


# Main entry point
def main() -> int:
    cfg = load_config()

    if not Path(cfg.shape_predictor).exists():
        print(f"Face detector '{Path(cfg.shape_predictor).absolute()}' could not be found. "
              f"Make sure to download the file from the link in the README and place it in the same directory as "
              f"'main.py'.", file=sys.stderr)
        return -1

    shape_predictor = dlib.shape_predictor(cfg.shape_predictor)

    # File management
    if Path(cfg.output_error_dir).exists():
        shutil.rmtree(cfg.output_error_dir)
    if Path(cfg.output_final_dir).exists():
        shutil.rmtree(cfg.output_final_dir)

    Path(cfg.input_dir).mkdir(exist_ok=True)
    Path(cfg.output_cache_dir).mkdir(exist_ok=True)
    Path(cfg.output_error_dir).mkdir(exist_ok=True)
    Path(cfg.output_final_dir).mkdir(exist_ok=True)

    temp_dir = TemporaryDirectory(prefix="facemation-")

    # Validation
    if len(glob.glob(f"{cfg.input_dir}/*.jpg")) == 0:
        print(f"No images detected in '{Path(cfg.input_dir).absolute()}'. "
              f"Are you sure you put them in the right place?",
              file=sys.stderr)
        return -1

    # Pre-process
    image_dates = {}
    eyes_left = {}
    eyes_right = {}

    pbar = tqdm(natsorted(glob.glob(f"{cfg.input_dir}/*.jpg")), desc="Pre-processing")
    for idx, input_file in enumerate(pbar):
        try:
            image_dates[idx] = cfg.filename_to_date(Path(input_file).stem)
        except Exception as exception:
            pbar.close()
            print_exception(exception, file=sys.stderr)
            print("\nFailed to parse date from filename. Check if 'filename_to_date' in your configuration matches the "
                  "filenames in the input directory.", file=sys.stderr)
            return -1

        image_hash = sha256sum(input_file)
        image_cache_file = Path(f"{cfg.output_cache_dir}/{image_hash}")

        # Read from cache if applicable
        if image_cache_file.exists():
            eyes_both = np.loadtxt(image_cache_file)
            eyes_left[input_file] = eyes_both[0]
            eyes_right[input_file] = eyes_both[1]
            continue

        # Detect eyes
        image_cv2 = cv2.imread(input_file)
        image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        detector = dlib.get_frontal_face_detector()
        detections = detector(image, 1)

        faces = dlib.full_object_detections()
        for detection in detections:
            faces.append(shape_predictor(image, detection))

        # Determine what to do if there are multiple faces
        if len(faces) > 1:
            if input_file in cfg.face_selection_override:
                face = sorted(list(faces), key=cfg.face_selection_override[Path(input_file).stem])[0]
            else:
                bb = [it.rect for it in faces]
                bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
                for it in bb:
                    image_cv2 = cv2.rectangle(image_cv2, it[0], it[1], (255, 0, 0), 5)
                cv2.imwrite(f"{cfg.output_error_dir}/{os.path.basename(input_file)}", image_cv2)

                print(
                    f"Too many faces: Found {len(faces)} in '{input_file}'. "
                    f"The image has been stored in '{Path(cfg.output_error_dir).absolute()}' with squares drawn around "
                    f"all faces that were found. "
                    f"You can select which face should be used by adjusting the 'face_selection_override' option; "
                    f"see 'config_default.py' for more information.",
                    file=sys.stderr)
                return -1
        else:
            face = faces[0]

        # Store results
        eyes_left[input_file] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(42, 48)]), axis=0)
        eyes_right[input_file] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(36, 42)]), axis=0)
        # noinspection PyTypeChecker
        np.savetxt(image_cache_file, np.vstack([eyes_left[input_file], eyes_right[input_file]]))

    # Calculate statistics
    eyes_dist = {it: math.dist(eyes_left[it], eyes_right[it]) for it in eyes_left.keys()}
    eyes_dist_avg = np.mean(np.array(list(eyes_dist.values())))

    eyes_left_np = np.array(list(eyes_left.values()))
    eyes_right_np = np.array(list(eyes_right.values()))

    eyes_left_avg = np.mean(eyes_left_np, axis=0)
    eyes_right_avg = np.mean(eyes_right_np, axis=0)
    eyes_center_avg = np.mean([eyes_left_avg, eyes_right_avg], axis=0)

    # Translate, rotate, resize
    width_min, height_min = [1e99, 1e99]
    for idx, input_file in enumerate(tqdm(natsorted(glob.glob(f"{cfg.input_dir}/*.jpg")),
                                          desc="Translating, rotating, resizing")):
        image = cv2.imread(input_file)

        height, width = image.shape[:2]
        eye_left, eye_right = eyes_left[input_file], eyes_right[input_file]
        eye_center = np.mean([eye_left, eye_right], axis=0)

        # Draw debug information
        if cfg.enable_debug:
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
        scale = eyes_dist_avg / eyes_dist[input_file]
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Write
        cv2.imwrite(f"{temp_dir.name}/{idx}.jpg", image)

        # Store smallest image seen
        height_new, width_new = image.shape[:2]
        width_min, height_min = min(width_min, width_new), min(height_min, height_new)

    # Make dimensions even
    width_min = width_min if width_min % 2 == 0 else width_min - 1
    height_min = height_min if height_min % 2 == 0 else height_min - 1

    # Crop and add text
    pbar = tqdm(natsorted(glob.glob(f"{temp_dir.name}/*.jpg")), desc="Cropping and adding text")
    for idx, input_file in enumerate(pbar):
        image = cv2.imread(input_file)
        height, width = image.shape[:2]

        # Crop
        width_excess = width - width_min
        width_start = int(width_excess / 2)
        width_end = width_start + width_min

        height_excess = height - height_min
        height_start = int(height_excess / 2)
        height_end = height_start + height_min

        image = image[height_start:height_end, width_start:width_end]

        # Add text
        try:
            caption = cfg.date_to_caption(image_dates[idx])
        except Exception as exception:
            pbar.close()
            print_exception(exception, file=sys.stderr)
            print("\nFailed to convert date to caption. Your 'filename_to_date' has been configured wrongly. Check your"
                  "configuration for more details.", file=sys.stderr)
            return -1

        image = write_on_image(image, caption, (0.05, 0.95), 0.05)

        # Write
        cv2.imwrite(f"{cfg.output_final_dir}/{idx}.jpg", image)


if __name__ == "__main__":
    main()
