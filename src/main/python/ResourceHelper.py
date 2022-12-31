import os
import sys


def exe_relative_path(relative_path: str) -> str:
    """
    Returns the absolute path to the file at [relative_path] relative to the Facemation executable.

    :param relative_path: the path relative to the Facemation executable
    :return: the absolute path to the file at [relative_path] relative to the Facemation executable
    """

    if getattr(sys, "frozen", False):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)

    return os.path.join(os.path.abspath(base_path), relative_path)


def cwd_relative_path(relative_path: str) -> str:
    """
    Returns the absolute path to the file at [relative_path] relative to the current working directory.

    :param relative_path: the path relative to the current working directory
    :return: the absolute path to the file at [relative_path] relative to the current working directory
    """

    return os.path.join(os.getcwd(), relative_path)


def resource_path(relative_path: str) -> str:
    """
    Returns the absolute path to the resource at [relative_path] which is bundled with Facemation by PyInstaller.

    :param relative_path: the relative path to the resource
    :return: the absolute path to the resource at [relative_path] which is bundled with Facemation by PyInstaller
    """

    if getattr(sys, "frozen", False):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
