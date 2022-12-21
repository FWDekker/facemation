# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, and crops all frames so that your eyes are aligned in each photo.

## Requirements
* Python 3.9 or newer
* [`venv`](https://docs.python.org/3/tutorial/venv.html) (e.g. `apt install python3-venv`)
* [`cmake`](https://cmake.org/) (e.g. `apt install cmake`) (required for `dlib` dependency)
* [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`) (to demux frames into a video)
* [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (in the same directory as `main.py`)

## How to use
1. Put your images in the `input/` folder.
   Files will be processed in alphabetical order.
2. Open a shell in the directory containing `main.py`.
3. (_Required once_) Create a [venv](https://docs.python.org/3/tutorial/venv.html):
   ```shell
   python3 -m venv venv/
   ```
4. Activate the venv:
   ```shell
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux
   ```
5. (_Required once_) Install dependencies:
   ```shell
   python3 -m pip install -r requirements.txt
   ```
6. (_Optional_) Configure the script by editing `config.py`.
   Check `config_default.py` for more information.
7. Run the script:
   ```shell
   python3 -m main
   ```
8. Run FFmpeg on the `output/final/` folder to combine the created frames into a video.
   For example:
   ```shell
   cd output/final/
   ffmpeg -f image2 -r 24 -i %d.jpg -vcodec libx264 -crf 24 out.mp4
   ```
   Tips:
   * You can add `-vf "transpose=2"` before `out.mp4` to rotate.
     See also [this StackOverflow answer](https://stackoverflow.com/a/9570992).
   * You can add `-vf "tpad=stop_mode=clone:stop_duration=3"` before `out.mp4` to freeze the last frame for 3 seconds.
   * You can add `-vf "minterpolate=fps=96:mi_mode=blend" ` before `out.mp4` to morph frames to make it look smoother,
     where `fps=96` changes the FPS to 96.
   * If you have two filters `-vf="filter1"` and `-vf="filter2"`, combine them as `-vf="filter1, filter2"`.
