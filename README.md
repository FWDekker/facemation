# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, crops, and captions all frames so your eyes are aligned in each photo, and
compiles these frames into a timelapse.

## Requirements
* (_Optional_) [FFmpeg](https://ffmpeg.org/) (can be disabled in `config.py`, see below)

## How to use
1. Check that you satisfy all the above requirements.
2. [Download the latest version of Facemation.](https://github.com/FWDekker/facemation/releases/latest)
   Unzip the downloaded archive.
3. Put your images in the `input/` folder.
   Files are processed in [natural sort order](https://en.wikipedia.org/wiki/Natural_sort_order).
4. (_Optional_) Configure Facemation by editing `config.py`.
   Check [`config_default.py`](https://github.com/FWDekker/facemation/blob/master/src/main/resources/config_default.py)
   for a list of all options.
5. Run the downloaded Facemation executable.
6. Check the output and adjust the configuration as desired.
   All intermediate results are heavily cached, so subsequent runs are much faster.

## Development
### Requirements
* [Python 3.9](https://www.python.org/)
* [venv](https://docs.python.org/3/tutorial/venv.html)
* [CMake](https://cmake.org/) (required to build `dlib`)
* C++ compiler (required to build `dlib`)
* [shape_predictor_5_face_landmarks.dat](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
  (store in `src/main/python/resources/`)

### Setup
1. Check that you satisfy all the above requirements.
   Depending on your configuration, you may need to run Python with `python` instead of `python3`.
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
   # Linux
   pyinstaller -y --clean -F --add-data="src/main/resources/*:." src/main/python/facemation.py
   cp src/main/resources/config_empty.py dist/config.py
   pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES
   mkdir dist/input/
   python3 -m zipfile -c "facemation-<system>-<version>.zip" dist/*
   # Windows
   # TODO
   ```
2. Run executable:
   ```shell
   # Linux
   dist/facemation
   # Windows
   dist/facemation.exe
   ```
