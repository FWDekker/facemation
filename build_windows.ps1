$ErrorActionPreference = "Stop"

if (Test-Path venv_build/) {Remove-Item -Recurse -Force venv_build/}
python -m venv venv_build/
if (-not $?) {throw "Failed to create venv."}
./venv_build/Scripts/activate
if (-not $?) {throw "Failed to activate venv."}
python -m pip install --upgrade pip wheel
if (-not $?) {throw "Failed to install requirements."}
python -m pip install -r requirements/build_windows.txt
if (-not $?) {throw "Failed to install requirements."}

if (Test-Path build/) {Remove-Item -Recurse -Force build/}
if (Test-Path dist/) {Remove-Item -Recurse -Force dist/}
if (Test-Path facemation.spec) {Remove-Item facemation.spec}
New-Item -Name "dist/" -ItemType "directory"
New-Item -Name "dist/input/" -ItemType "directory"
Copy-Item README.md -Destination dist/README.txt
Copy-Item src/main/resources/config_empty.py -Destination dist/config.py
pip-licenses --with-license-file --no-license-path --format=plain-vertical --output-file=dist/THIRD_PARTY_LICENSES.txt
if (-not $?) {throw "Failed to extract licenses from dependencies."}
Add-Content -Path dist/THIRD_PARTY_LICENSES.txt -Value $(Get-Content -Path src/main/resources/licenses_extra.txt)
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
