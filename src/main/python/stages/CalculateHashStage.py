import sys
from typing import List

from tqdm import tqdm

import Hasher
from Pipeline import Frame, Stage


class CalculateHashStage(Stage):
    """
    Reads simple image metadata.
    """

    def process(self, frames: List[Frame]) -> List[Frame]:
        """
        For each frame in [frames], writes the input image's hash into key `"hash"`.

        :param frames: the frames to process
        :return: the processed frames
        """

        for frame in tqdm(frames, desc="Calculating image hashes", file=sys.stdout):
            frame["hash"] = Hasher.hash_file(frame["path"])

        return frames
