import glob
import shutil
import sys
from pathlib import Path

from Cache import ImageCache, NdarrayCache
from CaptionStage import add_captions
from ConfigHelper import load_config
from DemuxStage import demux_images
from FindFacesStage import find_all_faces
from NormalizeStage import normalize_images
from ReadInputsStage import read_image_data
from UserException import UserException


def main() -> None:
    """
    Main entry point.

    :return: `None`
    """

    cfg = load_config()

    # Clean up from previous runs
    if Path(cfg["error_dir"]).exists():
        shutil.rmtree(cfg["error_dir"])
    if Path(cfg["frames_dir"]).exists():
        shutil.rmtree(cfg["frames_dir"])
    Path(cfg["output_path"]).unlink(missing_ok=True)

    Path(cfg["input_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["error_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["frames_dir"]).mkdir(parents=True, exist_ok=True)

    # Validate requirements and inputs
    if cfg["demux"]["enabled"] and shutil.which("ffmpeg") is None:
        print(f"FFmpeg is enabled in your configuration but is not installed. "
              f"Without FFmpeg, Facemation can create frames, but cannot produce a video. "
              f"Install FFmpeg or disable FFmpeg in your configuration. "
              f"Check the README for more information.", file=sys.stderr)
        return

    jpg_input_count = len(glob.glob(f"{cfg['input_dir']}/*.jpg"))
    all_input_count = len(glob.glob(f"{cfg['input_dir']}/*.*"))
    if jpg_input_count == 0:
        print(f"No images detected in '{Path(cfg['input_dir']).absolute()}'. "
              f"Are you sure you put them in the right place?",
              file=sys.stderr)
        return
    if jpg_input_count != all_input_count:
        print(f"Found {all_input_count - jpg_input_count} image(s) with an extension other than .jpg in "
              f"'{Path(cfg['input_dir']).absolute()}'. "
              f"Non-JPG images are not supported by Facemation. "
              f"Remove all files with an extension other than .jpg from the input folder.")
        return

    # Run facemation
    try:
        face_cache = NdarrayCache(cfg["cache_dir"], "face", ".cache")
        normalized_cache = ImageCache(cfg["cache_dir"], "normalized", ".jpg")
        captioned_cache = ImageCache(cfg["cache_dir"], "captioned", ".jpg")

        # Pre-process
        imgs = read_image_data(cfg["input_dir"])
        find_all_faces(imgs, face_cache, cfg["face_selection_override"], cfg["error_dir"])

        # Process
        normalize_images(imgs, face_cache, normalized_cache)

        # Post-process
        if cfg["caption"]["enabled"]:
            add_captions(imgs, normalized_cache, captioned_cache, cfg["caption"]["generator"])

        # Demux
        demux_input_cache = captioned_cache if cfg["caption"]["enabled"] else normalized_cache
        demux_images(imgs, demux_input_cache, cfg["frames_dir"], cfg["output_path"], cfg["demux"])

        print("Done!")
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)


if __name__ == "__main__":
    main()
