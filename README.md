# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, and crops all frames so that your eyes are aligned in each photo.

## Requirements
* Python 3.9 or newer
* `venv` (e.g. `apt install python3-venv`)
* `cmake` (e.g. `apt install cmake`) (required for `dlib` dependency)
* [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`) (to demux frames into a video)
* [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (in the same directory as `main.py`)

## How to use
1. Put your images in the `input/` folder.
   Files will be processed in alphabetical order.
2. (_Optional_) Configure the script by editing `config.py`.
3. Open a shell in the directory containing `main.py`.
4. (_Required once_) Create a [venv](https://docs.python.org/3/tutorial/venv.html):
   ```shell
   python3 -m venv venv/
   ```
5. Activate the venv:
   ```shell
   source venv/bin/activate
   ```
6. (_Required once_) Install dependencies:
   ```shell
   python3 -m pip install -r requirements.txt
   ```
7. Run the script:
   ```shell
   python3 -m main
   ```

   If multiple faces are detected, the script will fail and the violating image will be stored in `output/error/`, with squares drawn around all detected faces.
   You can manually select which face should be used by configuring the `face_selection_override` option in `config.py`, similar to the one in `config_default.py`.
   The documentation in `config_default.py` explains the workings in more detail.
8. Run FFmpeg on the `output/final/` folder to combine the created frames into a video.
   For example:
   ```shell
   cd output/final/
   ffmpeg -f image2 -r 24 -i %d.jpg -vcodec libx264 -crf 24 out.mp4
   ```
   * You can add `-vf "transpose=2"` to rotate.
   * You can add `-vf "tpad=stop_mode=clone:stop_duration=3"` to freeze the last frame for 3 seconds.
