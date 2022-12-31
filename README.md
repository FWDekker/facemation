# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, crops, and captions all frames so your eyes are aligned in each photo, and
compiles these frames into a timelapse.

## Requirements
* [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`)

## How to use
1. Check that you satisfy all the above requirements.
2. [Download the latest version of Facemation.](https://github.com/FWDekker/facemation/releases/latest)
   Unzip the downloaded archive.
3. Put your images in the `input/` folder.
   Files are processed in [natural sort order](https://en.wikipedia.org/wiki/Natural_sort_order).
4. (_Optional_) Configure Facemation by editing `config.py`.
   Check [`config_default.py`](https://github.com/FWDekker/facemation/blob/master/src/main/python/config_default.py) for
   a list of all options.
5. Run the downloaded Facemation executable.
6. Check the output and adjust the configuration as desired.
   All intermediate results are heavily cached, so subsequent runs are much faster.

## Development
### Requirements
* Python 3.9 or newer
* [`venv`](https://docs.python.org/3/tutorial/venv.html) (e.g. `apt install python3-venv`)
* [`cmake`](https://cmake.org/) (e.g. `apt install cmake`) (required to install `dlib` dependency)
* [FFmpeg](https://ffmpeg.org/) (e.g. `apt install ffmpeg`) (to demux frames into a video)
* [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (in your working directory)

### Setup
1. Check that you satisfy all the above requirements.
2. (_Required once_) Create a [venv](https://docs.python.org/3/tutorial/venv.html):
   ```shell
   python3 -m venv venv/
   ```
3. Activate the venv:
   ```shell
   # Linux
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```
4. (_Required once_) Install dependencies:
   ```shell
   python3 -m pip install -r requirements.txt
   ```
5. (_Optional_) Copy `src/main/python/config_empty.py` to `config_dev.py` in your working directory.
   `config_dev.py` overrides both `config_default.py` and `config.py`.

### Execute
Run the script:
```shell
cd src/main/python/
python3 -m facemation
```

### Build
1. Build executable into `dist/`:
   ```shell
   pyinstaller -y --clean -F --add-data="shape_predictor_68_face_landmarks.dat:." src/main/python/facemation.py
   cp src/main/python/config_empty.py dist/config.py
   pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES
   python3 -m zipfile -c "facemation-<system>-<version>.zip" dist/*
   ```
2. Run executable:
   ```shell
   # Linux
   dist/facemation
   # Windows
   dist/facemation.exe
   ```
