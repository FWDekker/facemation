$ErrorActionPreference = "Stop"

if (Test-Path venv_build/) {rm -r -force venv_build/}
python -m venv venv_build/
if (-not $?) {throw "Failed to create venv."}
./venv_build/Scripts/activate
if (-not $?) {throw "Failed to activate venv."}
python -m pip install --upgrade pip wheel
if (-not $?) {throw "Failed to install requirements."}
python -m pip install -r requirements/build_windows.txt
if (-not $?) {throw "Failed to install requirements."}

if (Test-Path build/) {rm -r -force build/}
if (Test-Path dist/) {rm -r -force dist/}
if (Test-Path facemation.spec) {rm facemation.spec}
mkdir dist/
mkdir dist/input/
cp README.md dist/README.txt
cp src/main/resources/config_empty.py dist/config.py
pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES.txt
if (-not $?) {throw "Failed to extract licenses from dependencies."}
pyinstaller -y -F `
    --noupx `
    --add-binary="src/main/resources/*.dll;." `
    --add-data="src/main/resources/config_default.py;." `
    --add-data="src/main/resources/Roboto-Regular.ttf;." `
    --add-data="src/main/resources/shape_predictor_5_face_landmarks.dat;." `
    src/main/python/facemation.py
if (-not $?) {throw "Failed to create executable."}

python -m zipfile -c "facemation-windows-$(cat version).zip" $(Resolve-Path -Relative "dist/*")
if (-not $?) {throw "Failed to ZIP distribution."}

deactivate
