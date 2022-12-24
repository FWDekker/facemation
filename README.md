# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, crops, and captions all frames so your eyes are aligned in each photo, and
compiles these frames into a timelapse.

## Requirements
* Python 3.9 or newer
* [`venv`](https://docs.python.org/3/tutorial/venv.html) (e.g. `apt install python3-venv`)
* [`cmake`](https://cmake.org/) (e.g. `apt install cmake`) (required for `dlib` dependency)
* [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`) (to demux frames into a video)
* [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (in the same directory as `main.py`)

## How to use
1. Put your images in the `input/` folder.
   Files are processed in [natural sort order](https://en.wikipedia.org/wiki/Natural_sort_order).
2. Open a shell in the directory containing `main.py`.
3. (_Required once_) Create a [venv](https://docs.python.org/3/tutorial/venv.html):
   ```shell
   python3 -m venv venv/
   ```
4. Activate the venv:
   ```shell
   # Windows
   .\venv\Scripts\activate
   # Linux
   source venv/bin/activate
   ```
5. (_Required once_) Install dependencies:
   ```shell
   python3 -m pip install -r requirements.txt
   ```
6. (_Optional_) Copy `config_default.py` to `config.py`, and adjust settings as desired.
   Settings are loaded from `config_default.py` and then overwritten by `config.py`.
7. Run the script:
   ```shell
   python3 -m main
   ```
8. Check the output and adjust the configuration as desired.
   All intermediate results are heavily cached, so subsequent runs are much faster.
