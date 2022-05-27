from datetime import datetime


config = {
    # Set to `True` to visualize debug information in the output.
    "enable_debug": False,
    # Predictor to use for finding facial features. You probably don't need to change this.
    "shape_predictor": "shape_predictor_68_face_landmarks.dat",

    # (Relative) directory to find the original frames in.
    "input_dir": "input/",
    # (Relative) directory to cache facial features in.
    "output_cache_dir": "output/cache/",
    # (Relative) directory to store images in that caused an error.
    "output_error_dir": "output/error/",
    # (Relative) directory to store the final processed images in.
    "output_final_dir": "output/final/",
    # (Relative) directory to store images in that are currently being processed.
    # TODO: Move to a real temp dir
    "output_temp_dir": "output/temp/",

    # Converts the filename of an image to the date on which it was taken.
    # TODO: Document these clearly
    "filename_to_date":
        (lambda it: datetime.strptime(it if it.count("_") == 2 else it[:-4], "IMG_%Y%m%d_%H%M%S").date()),
    # Converts the date of an image to an appropriate caption.
    "date_to_caption":
        (lambda it: f"Day {(it - datetime(year=2021, month=12, day=23).date()).days}"),

    # Selects the face to use when multiple faces are found in an image.
    "face_selection_override": {"IMG_20220112_124422": (lambda it: it.rect.top())}
}
