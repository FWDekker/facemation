#!/bin/bash
rm -rf venv_build/
python -m venv venv_build/
source venv_build/bin/activate
python -m pip install -r requirements.txt

rm -rf build/ dist/ facemation.spec
mkdir dist/
mkdir dist/input/
cp README.md dist/README.txt
cp src/main/resources/config_empty.py dist/config.py
pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES.txt
pyinstaller -y -F --add-data="src/main/resources/*:." src/main/python/facemation.py
staticx dist/facemation dist/facemation

python -m zipfile -c "facemation-linux-$(cat version).zip" dist/*
