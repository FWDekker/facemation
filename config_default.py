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

    # TODO: Document these clearly
    # Converts the filename of an image to the date on which it was taken.
    "filename_to_date": (lambda it: datetime.strptime("IMG_%Y%m%d_%H%M%S", it).date()),
    # Converts the date of an image to an appropriate caption.
    "date_to_caption": (lambda it: it.strftime("%Y-%m-%d")),
    # Selects the face to use when multiple faces are found in an image.
    # It's probably easiest to configure this each time the script fails, rather than pre-emptively filling it in.
    "face_selection_override": {f"example_1": (lambda it: it.rect.top()),  # Selects the highest face
                                f"example_2": (lambda it: -it.rect.top()),  # Selects the lowest face
                                f"example_3": (lambda it: it.rect.left()),  # Selects the left-most face
                                f"example_4": (lambda it: -it.rect.left())}  # Selects the right-most
}
