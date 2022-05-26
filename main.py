import glob
import hashlib
import math
import os
import shutil
from pathlib import Path

import cv2
import dlib
import imutils
import numpy as np
from tqdm import tqdm


# Settings
enable_debug = False  # Visualizes information in the output frames

input_dir = "input/"
output_dir = "output/"
output_cache_dir = "output/cache/"
output_error_dir = "output/error/"
output_final_dir = "output/final/"
output_temp_dir = "output/temp/"

face_selection_override = {
    f"{input_dir}IMG_20220112_124422.jpg": (lambda it: it.rect.top()),
}

shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Helper function
# Taken from https://stackoverflow.com/a/44873382
def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


if __name__ == "__main__":
    # Delete old files
    if Path(output_error_dir).exists():
        shutil.rmtree(output_error_dir)
    if Path(output_final_dir).exists():
        shutil.rmtree(output_final_dir)
    if Path(output_temp_dir).exists():
        shutil.rmtree(output_temp_dir)

    Path(output_dir).mkdir(exist_ok=True)
    Path(output_cache_dir).mkdir(exist_ok=True)
    Path(output_error_dir).mkdir(exist_ok=True)
    Path(output_final_dir).mkdir(exist_ok=True)
    Path(output_temp_dir).mkdir(exist_ok=True)

    # Pre-process
    eyes_left = {}
    eyes_right = {}
    for idx, input_file in enumerate(tqdm(sorted(glob.glob(f"{input_dir}/*.jpg")), desc="Pre-processing")):
        image_hash = sha256sum(input_file)
        image_cache_file = Path(f"{output_cache_dir}/{image_hash}")

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
            if input_file in face_selection_override:
                face = sorted(list(faces), key=face_selection_override[input_file])[0]
            else:
                bb = [it.rect for it in faces]
                bb = [((it.left(), it.top()), (it.right(), it.bottom())) for it in bb]
                for it in bb:
                    image_cv2 = cv2.rectangle(image_cv2, it[0], it[1], (255, 0, 0), 5)
                cv2.imwrite(f"{output_error_dir}/{os.path.basename(input_file)}", image_cv2)

                raise Exception(f"Too many faces: Found {len(faces)} in '{input_file}'. "
                                f"See also file in '{output_error_dir}'.")
        else:
            face = faces[0]

        # Store results
        eyes_left[input_file] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(42, 48)]), axis=0)
        eyes_right[input_file] = np.mean(np.array([(face.part(i).x, face.part(i).y) for i in range(36, 42)]), axis=0)
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
    for idx, input_file in enumerate(tqdm(sorted(glob.glob(f"{input_dir}/*.jpg")),
                                          desc="Translating, rotating, resizing")):
        image = cv2.imread(input_file)

        height, width = image.shape[:2]
        eye_left, eye_right = eyes_left[input_file], eyes_right[input_file]
        eye_center = np.mean([eye_left, eye_right], axis=0)

        # Draw debug information
        if enable_debug:
            image = cv2.circle(image, eye_left.astype(int), radius=20, color=(0, 255, 0), thickness=-1)
            image = cv2.circle(image, eye_right.astype(int), radius=20, color=(0, 255, 0), thickness=-1)
            image = cv2.circle(image, eyes_left_avg.astype(int), radius=20, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, eyes_right_avg.astype(int), radius=20, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, eye_center.astype(int), radius=20, color=(255, 0, 0), thickness=-1)

        # Translate
        eyes_center = np.mean([eye_left, eye_right], axis=0)
        translation = [eyes_center_avg[0] - eyes_center[0], eyes_center_avg[1] - eyes_center[1]]
        T = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        image = cv2.warpAffine(image, T, (width, height))

        # Rotate
        eye_right_relative = eye_right - eyes_center
        angle = math.atan(eye_right_relative[1] / eye_right_relative[0])
        image = imutils.rotate(image, angle)

        # Resize
        scale = eyes_dist_avg / eyes_dist[input_file]
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Write
        cv2.imwrite(f"{output_temp_dir}/{idx}.jpg", image)

        # Store smallest image seen
        height_new, width_new = image.shape[:2]
        width_min, height_min = min(width_min, width_new), min(height_min, height_new)

    # Make dimensions even
    width_min = width_min if width_min % 2 == 0 else width_min - 1
    height_min = height_min if height_min % 2 == 0 else height_min - 1

    # Reshape
    for idx, input_file in enumerate(tqdm(sorted(glob.glob(f"{output_temp_dir}/*.jpg")), desc="Reshaping")):
        image = cv2.imread(input_file)
        height, width = image.shape[:2]

        width_excess = width - width_min
        width_start = int(width_excess / 2)
        width_end = width_start + width_min

        height_excess = height - height_min
        height_start = int(height_excess / 2)
        height_end = height_start + height_min

        image = image[height_start:height_end, width_start:width_end]
        cv2.imwrite(f"{output_final_dir}/{idx}.jpg", image)
