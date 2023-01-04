# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, crops, and captions all frames so your eyes are aligned in each photo, and
compiles these frames into a timelapse.

## Installation
### Windows
1. [Download FFmpeg.](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
   (If you have [7-Zip](https://www.7-zip.org/) installed,
   [download the `.7z` archive instead](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z).)
2. Unzip the downloaded archive into a new directory.
3. [Download the latest version of Facemation for Windows.](https://github.com/FWDekker/facemation/releases/latest)
4. Unzip the downloaded archive into another new directory.
5. Enter the directory where you unzipped FFmpeg, enter the `bin` directory, and copy the file `ffmpeg.exe` to the
   directory where you unzipped Facemation.  
   You should now have `facemation.exe` and `ffmpeg.exe` in the same directory.

### Linux
1. Install [FFmpeg](https://ffmpeg.org/).
   On Ubuntu/Debian, you can simply do `apt install ffmpeg`.
2. [Download the latest version of Facemation for Linux.](https://github.com/FWDekker/facemation/releases/latest)
3. Unzip the downloaded archive into a new directory.

## How to use
1. Enter the directory where you installed Facemation.
2. Put the images you want Facemation to process in the `input` directory.
3. Rename files if necessary so that they are in the right order.
   Images are processed in [natural sort order](https://en.wikipedia.org/wiki/Natural_sort_order).
4. Execute `facemation` by double-clicking it.
5. Check the output and adjust `config.py` as desired.
   Check [`config_default.py`](https://github.com/FWDekker/facemation/blob/master/src/main/resources/config_default.py)
   for a list of all options.
   All intermediate results are heavily cached, so subsequent runs are much faster.

## Development
### Requirements
* [Python 3.10](https://www.python.org/) (invoked as `python`)
* [shape_predictor_5_face_landmarks.dat](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
  (store in `src/main/python/resources/`)
* [venv](https://docs.python.org/3/tutorial/venv.html)
* [CMake](https://cmake.org/) (required to build `dlib`)
* C++ compiler (required to build `dlib`)
  (Debian: `apt install g++`.
  Windows: [Visual C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/))

### Run script for development
#### Setup
1. Check that you satisfy the development requirements.
   On Windows, always use PowerShell.
2. Create a [venv](https://docs.python.org/3/tutorial/venv.html):
   ```shell
   python -m venv venv/
   ```
3. Activate the venv:
    * Linux
      ```shell
      source venv/bin/activate
      ```
    * Windows PowerShell
      ```shell
      ./venv/Scripts/activate
      ```
4. Install dependencies:
   ```shell
   python -m pip install -r requirements.txt
   ```
5. (_Optional_) Create `config_dev.py` to override both `config_default.py` and `config.py`.
   ```shell
   cp src/main/python/config_empty.py config_dev.py
   ```

#### Usage
1. Activate the venv:
    * Linux
      ```shell
      source venv/bin/activate
      ```
    * Windows PowerShell
      ```shell
      ./venv/Scripts/activate
      ```
2. Run script:
   ```shell
   python src/main/python/facemation.py
   ```

### Build distribution
#### Distribution requirements
* (_Linux only_) [Requirements for `staticx](https://staticx.readthedocs.io/en/latest/installation.html)

#### Usage
1. Check that you satisfy the development and the extra requirements.
   On Windows, always use PowerShell.
2. Check the version number in the `version` file.
3. Check that `config_empty.py` is up-to-date with `config_default.py`.
4. Build executable into `dist/` and create `.zip` distribution:
   * Linux
     ```shell
     ./build_linux.sh
     ```
   * Windows PowerShell
     ```shell
     ./build_windows.ps1
     ```
5. Run executable:
   ```shell
   dist/facemation
   ```
