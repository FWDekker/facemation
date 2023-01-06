# Facemation
Turn your daily selfies into a good-looking timelapse.

This script automatically scales, rotates, crops, and captions all frames so your eyes are aligned in each photo, and
compiles these frames into a timelapse.

## Usage instructions
If you need help with installing or using Facemation, please feel free to
[start a discussion](https://github.com/FWDekker/facemation/discussions) or
[contact me directly](https://fwdekker.com/about/).

### How to install
#### Windows 10 / Windows 11
1. [Download FFmpeg.](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
   (If you have [7-Zip](https://www.7-zip.org/) installed,
   [download the `.7z` archive instead](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z).)
2. Unzip the downloaded FFmpeg archive into a new directory.
3. [Download the latest version of Facemation for Windows.](https://github.com/FWDekker/facemation/releases/latest)
4. Unzip the downloaded Facemation archive into another new directory.
5. Enter the directory where you unzipped FFmpeg, enter the `bin` directory, and copy the file `ffmpeg.exe` to the
   directory where you unzipped Facemation.  
   You should now have `facemation.exe` and `ffmpeg.exe` in the same directory.

#### Linux
1. Install [FFmpeg](https://ffmpeg.org/).
   On Ubuntu/Debian, you can simply do `apt install ffmpeg`.
2. [Download the latest version of Facemation for Linux.](https://github.com/FWDekker/facemation/releases/latest)
3. Unzip the downloaded archive into a new directory.

#### macOS
Unfortunately, I don't have macOS, which means that I cannot create an executable for macOS systems.
Your best bet is probably to run the Python scripts directly by following the development instructions below, but even
then I cannot guarantee it will work.
If you have suggestions for how I can solve this, please let me know by
[opening an issue](https://github.com/FWDekker/facemation/issues),
[starting a discussion](https://github.com/FWDekker/facemation/discussions), or
[contacting me directly](https://fwdekker.com/about/).

### How to use
1. Enter the directory where you installed Facemation.
2. Put the images you want Facemation to process in the `input` directory.
3. Rename files if necessary so that they are in the right order.
   Images are processed in [natural sort order](https://en.wikipedia.org/wiki/Natural_sort_order).
4. Execute `facemation` by double-clicking it.
5. Check the created video in `output/facemation.mp4`.

All intermediate results are heavily cached, so subsequent runs are much faster.

### How to configure
You can change how Facemation behaves by editing the `config.py` file.
Below are some examples of how you can configure Facemation.
Check [`config_default.py`](https://github.com/FWDekker/facemation/blob/master/src/main/resources/config_default.py) for
a list of all options.

#### Disable FFmpeg
If you do not have FFmpeg, you can disable it.
Facemation will still work, but will skip the final step of creating a video.

```python
config = {
    "ffmpeg": {
        "enabled": False,
    }
}
```

#### Show the date in each frame
This code assumes that each filename is something like `IMG_20230104_174807.jpg`.

```python
from datetime import datetime

config = {
    "caption": {
        "enabled": True,
        "generator": (lambda filename: str(datetime.strptime(filename, "IMG_%Y%m%d_%H%M%S.jpg").date())),
    },
}
```

#### Show the number of days since an important event in each frame
This code assumes that each filename is something like `IMG_20230104_174807.jpg`.

```python
from datetime import datetime

important_date = datetime(year=2023, month=1, day=1).date()

config = {
    "caption": {
        "enabled": True,
        "generator":
            (lambda filename: str((datetime.strptime(filename, "IMG_%Y%m%d_%H%M%S.jpg").date() - important_date).days)),
    },
}
```

#### Adding music
Put your music file in the directory that contains `config.py`.
Then, update your configuration as below;
replace `music.mp3` with the name of your music file.
```python
config = {
    "ffmpeg": {
        "custom_inputs": ["-i", "music.mp3"],
        "custom_output_options": ["-map", "0:v", "-map", "1:a", "-shortest"],
    },
}
```

## Development instructions
If you are a developer and want to help with or change Facemation, these instructions are for you.

### Requirements
#### All systems
* [`shape_predictor_5_face_landmarks.dat`](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
  (extract and store in `src/main/python/resources/`)
* [`Roboto-Regular.ttf`](https://fonts.google.com/specimen/Roboto)
  (extract and store in `src/main/python/resources/`)

#### Linux
* [Python 3.10](https://www.python.org/)  
  The commands in this README invoke Python as `python`.
  Use `python3` instead if you have not linked `python` to `python3`.
* [venv](https://docs.python.org/3/tutorial/venv.html)  
  You can check if you have `venv` installed by running `python -m venv`;
  if you see usage information, you have `venv`.
  To install `venv` on Debian/Ubuntu, run `apt install python3-venv`.
* [CMake](https://cmake.org/) (required to build `dlib`)  
  On Debian/Ubuntu, install with `apt install cmake`.
* C++ compiler (required to build `dlib`)  
  On Debian/Ubuntu, install with `apt install g++`.

#### Windows 10/11
* Always use PowerShell.
* [Python 3.10](https://www.python.org/)
* [CMake](https://cmake.org/) (required to build `dlib`)
* C++ compiler (required to build `dlib`)  
  You will need either Visual Studio (an editor) or Visual Studio Tools (a library).
  You can find both on the [Visual Studio downloads page](https://visualstudio.microsoft.com/downloads/).

### Run script for development
#### Setup
1. Check that you satisfy the development requirements.
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
   python -m pip install --upgrade pip wheel
   python -m pip install -r requirements.txt
   ```
5. (_Optional_) Create `config_dev.py` to override both `config_default.py` and `config.py`.
   ```shell
   cp src/main/resources/config_empty.py config_dev.py
   ```
   Note that `config_dev.py` is always searched for in the current working directory.

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

### Build executable for distribution
#### Requirements
* All development requirements listed above
* (_Linux only_) [Requirements for `staticx`](https://staticx.readthedocs.io/en/latest/installation.html)
* (_Windows only_) Always use PowerShell
* (_Windows only_) [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)  
  Copy all DLLs in `C:/Program Files (x86)/Windows Kits/10/Redist/[version]/ucrt/x64` to `src/python/resources/`.
  Note that the `[version]` in the path differs per system.
  I don't know what the implications of this are.

#### Usage
1. Check that you satisfy the distribution requirements.
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

## Acknowledgements
In chronological order of contribution:
* Thanks to [Luc Everse](https://github.com/cmpsb) for finding a bunch of bugs in v1.0.0!

If I should add, remove, or change anything here, just open an issue or email me!
