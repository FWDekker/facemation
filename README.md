# Facemation
Turn your daily selfies into an animation.

Given a sequence of input photos, rotates and centres the photos, and outputs them as an animation.

## Requirements
- Python 3.9
- Install requirements from `requirements.txt`

## How to use
1. Put images in `input/` folder.
2. Rename photos to sequential numbering with
    ```bash
    $> ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done
    ```
3. Run this script.
4. Run ffmpeg on the `output/` folder.
