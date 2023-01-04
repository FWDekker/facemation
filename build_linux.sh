#!/bin/bash
set -e

rm -rf venv_build/
python3 -m venv venv_build/
source venv_build/bin/activate
python3 -m pip install --upgrade pip wheel
python3 -m pip install -r requirements/build_linux.txt

rm -rf build/ dist/ facemation.spec
mkdir dist/
mkdir dist/input/
cp README.md dist/README.txt
cp src/main/resources/config_empty.py dist/config.py
# TODO: Add licenses for Python and Roboto font
pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES.txt
pyinstaller -y -F \
  --noupx \
  --add-data="src/main/resources/config_default.py:." \
  --add-data="src/main/resources/Roboto-Regular.ttf:." \
  --add-data="src/main/resources/shape_predictor_5_face_landmarks.dat:." \
  src/main/python/facemation.py
staticx dist/facemation dist/facemation

python3 -m zipfile -c "facemation-linux-$(cat version).zip" dist/*

deactivate
