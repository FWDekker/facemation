import sys
from multiprocessing import freeze_support

from ConfigLoader import load_config
from Pipeline import Pipeline
from UserException import UserException
from stages.CaptionStage import CaptionStage
from stages.DemuxStage import DemuxStage
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
    pipeline.register(FindFacesStage(cfg["paths"]["cache"], cfg["paths"]["error"], cfg["face_selection_overrides"]))
    pipeline.register(NormalizeStage(cfg["paths"]["cache"]))
    if cfg["caption"]["enabled"]:
        pipeline.register(CaptionStage(cfg["paths"]["cache"], cfg["caption"]["generator"]))
    if cfg["demux"]["enabled"]:
        # TODO: Make `output` path relative to executable, not relative to `frame`
        pipeline.register(DemuxStage(cfg["paths"]["output"], cfg["demux"]))

    try:
        pipeline.run(cfg["paths"]["input"], cfg["paths"]["frames"])
        print("Done!")
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)


if __name__ == "__main__":
    freeze_support()
    main()
