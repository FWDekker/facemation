import glob
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, List

import numpy as np
from PIL import Image

import Files
from ImageLoader import load_image

"""The type of data stored in a cache."""
T = TypeVar("T")


# TODO: Keep track of `state`s in an index file, to prevent overly long filenames
# TODO: Add some kind of versioning, in case code makes a breaking change w.r.t. users' existing caches
class Cache(ABC, Generic[T]):
    """
    Stores data identified by a key, associated with a state that identifies the contents of the datum.

    Only a single datum can be stored under any given key. If a datum is stored under a used key, the existing datum and
    its state are overwritten.
    """

    directory: str
    prefix: str
    suffix: str

    def __init__(self, directory: str, prefix: str, suffix: str):
        """
        Constructs a new `Cache`.

        :param directory: the directory to store cached files in
        :param prefix: the string to prefix all cached files with
        :param suffix: the string to suffix all cached files with
        """

        Files.mkdir(directory)

        self.directory = directory
        self.prefix = prefix
        self.suffix = suffix

    def path(self, key: str, state: str = "") -> Path:
        """
        Returns the path to the data cached under [key] and [state].

        :param key: the key to search for
        :param state: the state to search for
        :return: the path to the data cached under [key] and [state]
        """

        return Path(f"{self.directory}/{self.prefix}-{key}-{state}{self.suffix}").resolve()

    def path_all(self, key: str) -> List[Path]:
        """
        Returns all paths to data cached under [key] and any state.

        :param key: the key to search for
        :return: all paths to data cached under [key] and any state
        """

        return [Path(it) for it in glob.glob(f"{self.directory}/{self.prefix}-{key}*{self.suffix}")]

    def has(self, key: str, state: str = "") -> bool:
        """
        Returns `True` if and only if a datum with [key] and [state] has been stored.

        :param key: the key to search for
        :param state: the state to search for
        :return: `True` if and only if a datum with [key] and [state] has been stored.
        """

        return self.path(key, state).exists()

    def load(self, key: str, state: str = "") -> T:
        """
        Returns the cached data associated with [key] and [state].

        :param key: the key to load data for
        :param state: the state to load data for
        :return: the cached data associated with [key] and [state]
        """

        path = self.path(key, state)
        if not path.exists():
            raise Exception(f"Tried to load '{path}' from cache '{self.prefix}', but no such file exists.")

        return self._read_data(path)

    def cache(self, data: T, key: str, state: str = "") -> Path:
        """
        Removes all other data cached under [key], and caches [data] under [key] and [state].

        :param data: the data to cache
        :param key: the key to cache [data] under
        :param state: the state to cache [data] under
        :return: the path under which [data] has been stored
        """

        if "-" in key:
            raise Exception(f"Key must not contain '-', but was '{key}'.")
        if "-" in state:
            raise Exception(f"State must not contain '-', but was '{state}'.")

        for path in self.path_all(key):
            path.unlink()

        path = self.path(key, state)
        self._write_data(path, data)

        return path

    @abstractmethod
    def _write_data(self, path: Path, data: T) -> None:
        """
        Serializes and writes [data] to [path].

        :param path: the path to write [data] to
        :param data: the data to write to [path]
        :return: `None`
        """

        pass

    @abstractmethod
    def _read_data(self, path: Path) -> T:
        """
        Reads data from [path] and returns the deserialized value.

        :param path: the path to read data from
        :return: the deserialized data read from [path]
        """

        pass


class ImageCache(Cache[Image.Image]):
    """
    Caches images from the Pillow library.
    """

    def _write_data(self, path: Path, data: Image.Image) -> None:
        data.save(path)

    def _read_data(self, path: Path) -> Image.Image:
        return load_image(path)


class NdarrayCache(Cache[np.ndarray]):
    """
    Caches arbitrary Numpy arrays.
    """

    def _write_data(self, path: Path, data: np.ndarray) -> None:
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_data(self, path: Path) -> np.ndarray:
        with open(path, "rb") as file:
            return pickle.load(file)
