if (Test-Path venv_build/) {rm -r -force venv_build/}
python -m venv venv_build/
./venv/Scripts/activate
python -m pip install -r requirements.txt

if (Test-Path build/) {rm -r -force build/}
if (Test-Path dist/) {rm -r -force dist/}
if (Test-Path facemation.spec) {rm facemation.spec}
mkdir dist/
mkdir dist/input/
cp README.md dist/README.txt
cp src/main/resources/config_empty.py dist/config.py
pyinstaller -y -F --add-data="src/main/resources/*;." src/main/python/facemation.py
pip-licenses --with-license-file --no-license-path --output-file=dist/THIRD_PARTY_LICENSES.txt

python -m zipfile -c "facemation-windows-$(cat version).zip" $(Resolve-Path -Relative "dist/*")
