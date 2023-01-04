import sys
from multiprocessing import freeze_support

from ConfigLoader import load_config
from Pipeline import Pipeline
from UserException import UserException
from stages.CaptionStage import CaptionStage
from stages.FfmpegStage import FfmpegStage
from stages.FindFacesStage import FindFacesStage
from stages.NormalizeStage import NormalizeStage
from stages.ReadMetadataStage import ReadMetadataStage


def main() -> None:
    """
    Main entry point.

    :return: `None`
    """

    cfg = load_config()

    pipeline = Pipeline()
    pipeline.register(ReadMetadataStage())
    pipeline.register(FindFacesStage(cfg["paths"]["cache"], cfg["paths"]["error"],
                                     cfg["find_faces"]["face_selection_overrides"]))
    pipeline.register(NormalizeStage(cfg["paths"]["cache"]))
    if cfg["caption"]["enabled"]:
        pipeline.register(CaptionStage(cfg["paths"]["cache"], cfg["caption"]["generator"]))
    if cfg["ffmpeg"]["enabled"]:
        pipeline.register(FfmpegStage(cfg["paths"]["output"], cfg["ffmpeg"]))

    try:
        pipeline.run(cfg["paths"]["input"], cfg["paths"]["frames"])
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)

    if getattr(sys, "frozen", False):
        input("Press Enter to close.")


if __name__ == "__main__":
    freeze_support()
    main()
