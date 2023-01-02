import sys

from Pipeline import Pipeline
from Config import load_config
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
    pipeline.register([
        ReadMetadataStage(),
        FindFacesStage(cfg["paths"]["cache"], cfg["paths"]["error"], cfg["face_selection_overrides"]),
        NormalizeStage(cfg["paths"]["cache"]),
        CaptionStage(cfg["paths"]["cache"], cfg["caption"]["generator"]),  # TODO: Remove this if captions disabled
        DemuxStage(cfg["paths"]["frames"], cfg["paths"]["output"], cfg["demux"])
    ])

    try:
        pipeline.run(cfg["paths"]["input"])
        print("Done!")
    except UserException as exception:
        print("Error: " + exception.args[0], file=sys.stderr)


if __name__ == "__main__":
    main()
