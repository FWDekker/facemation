import glob
import shutil
from pathlib import Path
from typing import List, Sequence


def cleardir(path: str) -> None:
    """
    Removes all files at [path] by removing the directory and then re-creating it.

    :param path: the path to the directory to empty
    :return: `None`
    """

    if Path(path).exists():
        shutil.rmtree(path)

    mkdir(path)


def mkdir(path: str) -> None:
    """
    Creates the directory at [path], including parents, and does not fail if the directory already exists.

    :param path: the path to the directory to create
    :return: `None`
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def rm(path: str) -> None:
    """
    Removes the file at [path], and does not fail if the file does not exist.

    :param path: the path to the file to delete
    :return: `None`
    """

    Path(path).unlink(missing_ok=True)


def glob_extensions(root_dir: str, extensions: str) -> List[str]:
    """
    Returns all files in [root_dir] with an extension in [extensions], akin to
    `glob.glob(root_dir + "/{" + extensions + "}")`.

    :param extensions: the extensions to filter, each formatted as `"*.jpg"`
    :param root_dir: the directory to glob in
    :return: all files in [root_dir] with an extension in [extensions]
    """

    extension_list = [f"*.{it}" for it in extensions.split(",")]
    return [file for files in [glob.glob(root_dir + "/" + it) for it in extension_list] for file in files]
