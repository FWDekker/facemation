import sys
from pathlib import Path


def cwd_relative_path(relative_path: str) -> Path:
    """
    Returns the path to the file at [relative_path] relative to the current working directory.

    :param relative_path: the path relative to the current working directory
    :return: the path to the file at [relative_path] relative to the current working directory
    """

    return Path.cwd().joinpath(relative_path)


def exe_relative_path(relative_path: str) -> Path:
    """
    Returns the path to the file at [relative_path] relative to the invoked executable.

    :param relative_path: the path relative to the Facemation executable
    :return: the path to the file at [relative_path] relative to the invoked executable
    """

    if getattr(sys, "frozen", False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent

    return base_path.joinpath(relative_path)


def resource_path(relative_path: str) -> Path:
    """
    Returns the path to the resource at [relative_path].

    A resource is a file bundled into the executable file by PyInstaller. If this function is not invoked from the
    executable, the returned path is relative to the current working directory.

    :param relative_path: the relative path to the resource
    :return: the path to the resource at [relative_path]
    """

    if getattr(sys, "frozen", False):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path.cwd()

    return base_path.joinpath(relative_path)
