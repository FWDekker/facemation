# This file contains the default configuration. To change the configuration, override settings in the `config.py` file,
# which must be in the same directory as the Facemation executable.
config = {
    # (Relative) directory to find the original frames in.
    "input_dir": "input/",
    # (Relative) directory to store images in that caused an error.
    "error_dir": "output/error/",
    # (Relative) directory to cache intermediate results in.
    "cache_dir": "output/cache/",
    # (Relative) directory to store final frames in.
    "frames_dir": "output/frames/",
    # (Relative) directory to store created video in, relative to `frames_dir`.
    "output_path": "../facemation.mp4",

    # Determines which face should be used for normalization if an image contains multiple faces.
    #
    # Add an entry for each image that contains multiple faces. The entry maps the filename of the image (without
    # directory, with extension) to a function that returns a number. Specifically, the function takes an instance of
    # `dlib.full_object_detection` (see also http://dlib.net/python/index.html#dlib_pybind11.full_object_detection), and
    # returns an `int`. When the given file is processed, this function is executed on each face in that image, and the
    # face with the lowest value will be used for all subsequent processing.
    #
    # The easiest way to configure overrides is based on the position of the face, as shown in the default examples
    # below. To find which face you need, first run the main script, and once it fails because it has detected multiple
    # faces, check the image in the `error_dir` directory and add an appropriate override function.
    "face_selection_override": {
        # Selects the face whose top edge is closest to the top of the image.
        f"example_top_top.jpg": (lambda it: it.rect.top()),
        # Selects the face whose bottom edge is closest to the top of the image.
        f"example_bottom_top.jpg": (lambda it: it.rect.bottom()),
        # Selects the face whose top edge is closest to the bottom of the image.
        f"example_top_bottom.jpg": (lambda it: -it.rect.top()),
        # Selects the face whose left edge is closest to the left of the image.
        f"example_left_left.jpg": (lambda it: it.rect.left()),
        # Selects the face whose left edge is closest to the right of the image.
        f"example_left_right.jpg": (lambda it: -it.rect.left()),
        # Selects the face whose top edge is closest to y = 500.
        f"example_top_500.jpg": (lambda it: abs(it.rect.top() - 500))
    },

    # Add text into the processed frames.
    "caption": {
        # Set to `True` to add a caption to each frame.
        "enabled": False,
        # Given the image's filename and the `Image` object from the PIL library, generates the string caption to add to
        # the image. See https://pillow.readthedocs.io/en/stable/reference/Image.html for more information on the
        # `Image` object.
        "generator": (lambda filename, image: f"{filename}: {image.size}"),
    },

    # Combine the processed images into a video.
    "demux": {
        # Set to `True` to automatically run FFmpeg at the end.
        "enabled": True,
        # The number of photos per second to show in the output video.
        "fps": "48",
        # The codec to use for the output video. x264 is a very widely supported codec.
        "codec": "libx264",
        # The "compression level" of the output video. A lower value means higher quality. Recommended between 18 and
        # 28.
        "crf": "23",
        # The video filters to apply when creating the output video.
        "video_filters": [
            # Pauses the first frame for 1 second at the start.
            "tpad=start_mode=clone:start_duration=1",
            # Pauses the last frame for 3 seconds at the end.
            "tpad=stop_mode=clone:stop_duration=3",
            # Morphs pictures into each other for a smoother transition effect.
            "minterpolate=fps=60:mi_mode=blend",
        ],
    }
}
