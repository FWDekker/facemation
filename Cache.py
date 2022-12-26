import glob
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TypeVar, Generic

import cv2
import numpy as np

T = TypeVar("T")


class Cache(ABC, Generic[T]):
    """
    Stores data identified by a key, associated with args that identify whether the file is up-to-date.
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
        Path(directory).mkdir(exist_ok=True)

        self.directory = directory
        self.prefix = prefix
        self.suffix = suffix

    def has(self, key: str, args: List[str]) -> bool:
        """
        Returns `True` if and only if a file with [key] and [args] has been stored.

        :param key: the key to search for
        :param args: the args to search for
        :return: `True` if and only if a file with [key] and [args] has been stored.
        """
        return Path(self.get_path(key, args)).exists()

    def has_any(self, key: str) -> bool:
        """
        Returns `True` if and only if any data with [key] and any args has been cached.

        :param key: the key to search for
        :return: `True` if and only if any data with [key] and any args has been cached
        """

        paths = glob.glob(self.get_glob(key))
        return len(paths) > 0

    def load(self, key: str, args: List[str]) -> T:
        """
        Returns the cached data associated with [key] and [args].

        :param key: the key to load data for
        :param args: the args to load data for
        :return: the cached data associated with [key] and [args]
        """

        path = self.get_path(key, args)

        if not Path(path).exists():
            raise Exception(f"Tried to load '{path}' from cache '{self.prefix}', but no such file exists.")

        return self._read_data(path)

    def load_any(self, key: str) -> T:
        """
        Returns any cached data associated with [key], regardless of `args`.

        Throws an [Exception] if no data has been cached under [key].

        :param key: the key to load data for
        :return: any cached data associated with [key], regardless of `args`
        """

        return self._read_data(self.get_path_any(key))

    def cache(self, key: str, args: List[str], data: T) -> None:
        """
        Caches [data] under [key] and [args], and removes all other data cached under [key].

        :param key: the key to cache [data] under
        :param args: the args to cache [data] under
        :param data: the data to cache
        :return: `None`
        """

        for file in glob.glob(self.get_glob(key)):
            Path(file).unlink()

        path = self.get_path(key, args)
        self._write_data(path, data)

    def get_path(self, key: str, args: List[str]) -> str:
        """
        Returns the path to the data cached under [key] and [args].

        :param key: the key to search for
        :param args: the args to search for
        :return: the path to the data cached under [key] and [args]
        """

        path = f"{self.directory}/{self.prefix}-{key}"
        if len(args) > 0:
            path += "-".join(args)
        path += self.suffix
        return path

    def get_path_any(self, key: str) -> str:
        """
        Returns the path to any data cached under [key] and any args.

        Throws an [Exception] if no data has been cached under [key].

        :param key: the key to search for
        :return: the path to any data cached under [key] and any args
        """

        paths = glob.glob(self.get_glob(key))
        if len(paths) == 0:
            raise Exception(f"Expected exactly 1 cached result for '{self.prefix}', but found 0.")

        return paths[0]

    def get_glob(self, key: str) -> str:
        """
        Returns the glob pattern that matches all data cached under [key].

        :param key: the key to search for
        :return: the glob pattern that matches all data cached under [key]
        """

        return f"{self.directory}/{self.prefix}-{key}*{self.suffix}"

    @abstractmethod
    def _write_data(self, path: str, data: T) -> None:
        """
        Serializes and writes [data] to the file at [path].

        :param path: the path to write [data] to
        :param data: the data to write to [path]
        :return: `None`
        """

        pass

    @abstractmethod
    def _read_data(self, path: str) -> T:
        """
        Reads data from the file at [path] and returns the deserialized value.

        :param path: the path to read data from
        :return: the deserialized data read from the file at [path]
        """

        pass


class ImageCache(Cache[np.ndarray]):
    """
    Caches images from the CV2 library.
    """

    def _write_data(self, path: str, data: np.ndarray) -> None:
        cv2.imwrite(path, data)

    def _read_data(self, path: str) -> np.ndarray:
        return cv2.imread(path)


class NdarrayCache(Cache[np.ndarray]):
    """
    Caches arbitrary Numpy arrays.
    """

    def _write_data(self, path: str, data: np.ndarray) -> None:
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_data(self, path: str) -> np.ndarray:
        with open(path, "rb") as file:
            return pickle.load(file)
