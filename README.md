# Facemation
Turn your daily selfies into an animation.

Given a sequence of input photos, rotates and centres the photos, and outputs them as an animation.

## Requirements
- Python 3.9
- Install requirements from `requirements.txt`
  - `dlib` requires `cmake` (e.g. `apt install cmake`)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`)

## How to use
1. Put images in `input/` folder.
2. Rename photos to sequential numbering with
    ```bash
    ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done
    ```
3. Run this script.

   If multiple faces are detected, add a face sorting function for the image to `face_selection_override`.
      The sorting function should return the desired face in the first position.
5. Run ffmpeg on the `output/` folder, for example:
   ```bash
   cd output/final/
   ffmpeg -f image2 -r 24 -i %d.jpg -vcodec libx264 -crf 24 out.mp4
   ```
   * You can add `-vf "transpose=2"` to rotate.
   * You can add `-vf "tpad=stop_mode=clone:stop_duration=3"` to freeze the last frame for 3 seconds.
