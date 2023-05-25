# This file contains the default configuration. To change the configuration, override settings in the `config.py` file,
# which must be in the same directory as the Facemation executable.
config = {
    # Locations of files. Usually, you don't need to change this.
    "pipeline": {
        # Directory with the original images to process and turn into a video.
        "input_dir": "input/",
        # Directory to cache intermediate results in to speed up future runs.
        "cache_dir": "output/cache/",
    },

    # Configure which face is selected in each image.
    "find_faces": {
        # Directory to store images in that caused an error.
        "error_dir": "output/error/",
        # Determines which face should be used for normalization if an image contains multiple faces.
        #
        # Add an entry for each image that contains multiple faces. The entry maps the filename of the image (without
        # directory, with extension) to a function that returns a number. Specifically, the function takes an instance
        # of `dlib.full_object_detection` (see also http://dlib.net/python/index.html#dlib_pybind11.full_object_
        # detection), and returns an `int`. When the given file is processed, the faces in that image are sorted
        # according to the function defined here for that image, and the face with the lowest value will be used for all
        # subsequent processing.
        #
        # The easiest way to configure overrides is based on the position of the face, as shown in the default examples
        # below. To find which face you need, first run the main script, and once it fails because it has detected
        # multiple faces, check the image in the `error_dir` directory and add an appropriate override function.
        "face_selection_overrides": {
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
            f"example_top_500.jpg": (lambda it: abs(it.rect.top() - 500)),
        },
    },

    # Add text into the processed frames.
    "caption": {
        # Set to `True` to add a caption to each frame.
        "enabled": False,
        # Generates the caption for the image using the image's filename (including extension).
        "generator": (lambda filename: filename),
    },

    # Combine the processed images into a video.
    # To learn more about FFmpeg, check https://ffmpeg.org/ or https://en.wikipedia.org/wiki/FFmpeg.
    "ffmpeg": {
        # Set to `True` to automatically run FFmpeg at the end. Set this to `False` if you do not have FFmpeg, or if you
        # prefer running FFmpeg with your own settings.
        "enabled": True,
        # The path to where FFmpeg is installed.
        "exe_path": "ffmpeg",
        # Directory to store final frames in.
        "frames_dir": "output/frames/",
        # Filename to store created video as.
        "output_path": "output/facemation.mp4",
        # The number of photos per second to show in the output video.
        "fps": 24,
        # The codec to use for the output video. x264 is a very widely supported codec.
        "codec": "libx264",
        # The "compression level" of the output video. A lower value means higher quality. Recommended between 18 and
        # 28.
        "crf": 23,
        # The video filters to apply when creating the output video.
        "video_filters": [
            # Pauses the first frame for 1 second at the start.
            "tpad=start_mode=clone:start_duration=1",
            # Pauses the last frame for 3 seconds at the end.
            "tpad=stop_mode=clone:stop_duration=3",
            # Morphs pictures into each other for a smoother transition effect.
            "minterpolate=fps=60:mi_mode=blend",
        ],
        # Additional global options to pass to FFmpeg.
        "custom_global_options": [],
        # Additional inputs to provide to FFmpeg.
        "custom_inputs": [],
        # Additional output options to provide to FFmpeg.
        "custom_output_options": [],
    }
}
