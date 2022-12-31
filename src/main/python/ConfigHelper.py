from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

from ResourceHelper import cwd_relative_path, exe_relative_path


def load_config() -> SimpleNamespace:
    """
    Loads config from `config.default.py` and overrides it with the config from `config_dev.py` if it exists.

    :return: the combined configuration
    """

    from config_default import config as cfg_default

    cfg_user_path = exe_relative_path("config.py")
    if Path(cfg_user_path).exists():
        cfg_user = SourceFileLoader("cfg_user", cfg_user_path).load_module().config
    else:
        cfg_user = {}

    cfg_dev_path = cwd_relative_path("config_dev.py")
    if Path(cfg_dev_path).exists():
        cfg_dev = SourceFileLoader("cfg_dev", cfg_dev_path).load_module().config
    else:
        cfg_dev = {}

    return SimpleNamespace(**(cfg_default | cfg_dev | cfg_user))
