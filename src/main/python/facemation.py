import sys
from multiprocessing import freeze_support

from ConfigLoader import load_config
from Pipeline import Pipeline
from UserException import UserException
from stages.CaptionStage import CaptionStage
from stages.FfmpegStage import FfmpegStage
from stages.FindFacesStage import FindFacesStage
from stages.NormalizeStage import NormalizeStage
from stages.CalculateHashStage import CalculateHashStage

error_info = f"If you think this is a bug, consider opening an issue at " \
             f"https://github.com/FWDekker/facemation/issues. " \
             f"If you need help, feel free to start a discussion at " \
             f"https://github.com/FWDekker/facemation/discussions, or contact the author directly at " \
             f"https://fwdekker.com/about/."


def excepthook(the_type, value, traceback):
    """
    Displays additional contact information when an exception goes uncaught.

    :param the_type: ignored
    :param value: ignored
    :param traceback: ignored
    :return: `None`
    """

    print(f"An unexpected error occurred while running Facemation:\n", file=sys.stderr)
    sys.__excepthook__(the_type, value, traceback)
    print(f"\n{error_info}", file=sys.stderr)


def main() -> None:
    """
    Main entry point.

    :return: `None`
    """

    cfg = load_config()

    pipeline = Pipeline()
    pipeline.register(CalculateHashStage())
    pipeline.register(FindFacesStage(cfg["find_faces"], cfg["pipeline"]["cache_dir"]))
    pipeline.register(NormalizeStage(0, 0, cfg["pipeline"]["cache_dir"]))
    if cfg["caption"]["enabled"]:
        pipeline.register(CaptionStage(0, 1, cfg["pipeline"]["cache_dir"], cfg["caption"]["generator"]))
    if cfg["ffmpeg"]["enabled"]:
        pipeline.register(FfmpegStage([0, 1], cfg["ffmpeg"]))

    pipeline.run(cfg["pipeline"]["input_dir"])

    if getattr(sys, "frozen", False):
        input("Press Enter to close.")


if __name__ == "__main__":
    try:
        sys.excepthook = excepthook

        if len(sys.argv) > 1:
            raise UserException("Facemation does not support command-line arguments. "
                                "To configure Facemation, edit the config.py file. "
                                "Check the README for more information.")

        freeze_support()
        main()
    except UserException as exception:
        print(f"Error: {exception.args[0]}\n\n{error_info}", file=sys.stderr)
        sys.exit(2)
