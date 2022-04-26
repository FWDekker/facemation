import glob
import math
import os
from pathlib import Path

import cv2
import imutils
import numpy as np


eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

input_dir = "input/"
output_dir = "output/"
Path(output_dir).mkdir(exist_ok=True)

input_files = os.listdir(input_dir)

# Pre-process images
print("# Pre-processing")
all_eyes = []
for input_file in glob.glob(f"{input_dir}/*.jpg"):
    print(input_file)
    image = cv2.imread(input_file)

    eyes = eye_cascade.detectMultiScale(image, 1.1, 4).tolist()
    eyes.sort(key=lambda it: it[2])  # Sort by size, last ones are largest
    eyes = eyes[-2:]
    eyes.sort(key=lambda it: it[0])  # Sort by x, so left eye is first
    all_eyes.append(eyes)
all_eyes = np.array(all_eyes)

avg_center = np.mean(np.mean(all_eyes[:, :, 0:2], axis=0), axis=0)
avg_distance = np.mean([math.sqrt(it[0] ** 2 + it[1] ** 2) for it in all_eyes[:, 1, 0:2] - avg_center])

# Translate, rotate, resize
print("# Translating, rotating, resizing")
max_width, max_height = [0, 0]
for input_file in glob.glob(f"{input_dir}/*.jpg"):
    print(input_file)

    image = cv2.imread(input_file)
    height, width = image.shape[:2]

    eyes = eye_cascade.detectMultiScale(image, 1.1, 4).tolist()
    eyes.sort(key=lambda it: it[2])  # Sort by size, last ones are largest
    eyes = eyes[-2:]
    eyes.sort(key=lambda it: it[0])  # Sort by x, so left eye is first

    # for (x, y, w, h) in eyes:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    eyes_center = np.mean(np.array(eyes)[:, 0:2], axis=0)
    translation = [avg_center[0] - eyes_center[0], avg_center[1] - eyes_center[1]]
    T = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    image = cv2.warpAffine(image, T, (width, height))

    right_eye_relative = np.array(eyes)[1, 0:2] + translation - eyes_center
    angle = math.atan(right_eye_relative[1] / right_eye_relative[0])
    image = imutils.rotate(image, angle)

    # distance = math.sqrt(right_eye_relative[0] ** 2 + right_eye_relative[1] ** 2)
    # scale = avg_distance / distance
    # image = cv2.resize(image, (int(width * scale), int(height * scale)))

    cv2.imwrite(f"{output_dir}/{os.path.basename(input_file)}", image)

    new_height, new_width = image.shape[:2]
    max_width, max_height = max(max_width, new_width), max(max_height, new_height)

max_width = max_width if max_width % 2 == 0 else max_width + 1
max_height = max_height if max_height % 2 == 0 else max_height + 1

# Make images same size
# print("Reshaping")
# for input_file in glob.glob(f"{output_dir}/*.jpg"):
#     image = cv2.imread(input_file)
#     height, width = image.shape[:2]
#
#     vert_border = max_height - height
#     top_border = int(vert_border)
#     bottom_border = int(vert_border) if vert_border == int(vert_border) else int(vert_border + 1)
#
#     hor_border = max_width - width
#     left_border = int(hor_border)
#     right_border = int(hor_border) if hor_border == int(hor_border) else int(hor_border + 1)
#
#     image = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT)
#     cv2.imwrite(f"{output_dir}/{os.path.basename(input_file)}", image)
