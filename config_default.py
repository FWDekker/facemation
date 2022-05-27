from datetime import datetime


# This file contains the default configuration. To change the configuration, either edit this file directly, or create
# a copy called `config.py` and adjust the values there. Values in `config.py` will always override those in this file.


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

    # Converts the filename of an image to the date on which it was taken.
    # See also https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    "filename_to_date": (lambda it: datetime.strptime("IMG_%Y%m%d_%H%M%S", it).date()),
    # Converts the date of an image to an appropriate caption.
    "date_to_caption": (lambda it: it.strftime("%Y-%m-%d")),
    # Determines which face should be used for normalization if an image contains multiple faces.
    #
    # Add an entry for each image that contains multiple faces. The entry maps the filename of the image (without
    # directory or extension) to a function that returns a number. Specifically, the function takes an instance of
    # `dlib.full_object_detection` (see also http://dlib.net/python/index.html#dlib_pybind11.full_object_detection), and
    # returns an `int`. When the given file is processed, this function is executed on each face in that image, and the
    # face with the lowest value will be used for all subsequent processing.
    #
    # The easiest way to configure overrides is based on the position of the face, as shown in the default examples
    # below. To find which face you need, first run the main script, and once it fails because it has detected multiple
    # faces, check the image in the `output_error_dir` directory and add an appropriate override function.
    "face_selection_override": {
        # Selects the face whose top edge is closest to the top of the image.
        f"example_top_top": (lambda it: it.rect.top()),
        # Selects the face whose bottom edge is closest to the top of the image.
        f"example_bottom_top": (lambda it: it.rect.bottom()),
        # Selects the face whose top edge is closest to the bottom of the image.
        f"example_top_bottom": (lambda it: -it.rect.top()),
        # Selects the face whose left edge is closest to the left of the image.
        f"example_left_left": (lambda it: it.rect.left()),
        # Selects the face whose left edge is closest to the right of the image.
        f"example_left_right": (lambda it: -it.rect.left()),
        # Selects the face whose top edge is closest to y = 500.
        f"example_top_500": (lambda it: abs(it.rect.top() - 500))}
}
