from importlib.machinery import SourceFileLoader
from typing import Any

from mergedeep import merge

import Resolver

FacemationConfig = Any


def load_config() -> FacemationConfig:
    """
    Loads config from `config_default.py`, overrides it with `config.py` (if it exists) in the same directory as the
    executable that was invoked, and then overrides that with `config_dev.py` (if it exists) in the current working
    directory.

    :return: the combined configuration
    """

    cfg_default_path = Resolver.resource_path("config_default.py")
    cfg_default = SourceFileLoader("cfg_default", str(cfg_default_path)).load_module().config

    cfg_user_path = Resolver.exe_relative_path("config.py")
    if cfg_user_path.exists():
        cfg_user = SourceFileLoader("cfg_user", str(cfg_user_path.resolve())).load_module().config
    else:
        cfg_user = {}

    cfg_dev_path = Resolver.cwd_relative_path("config_dev.py")
    if cfg_dev_path.exists():
        cfg_dev = SourceFileLoader("cfg_dev", str(cfg_dev_path.resolve())).load_module().config
    else:
        cfg_dev = {}

    return merge({}, cfg_default, cfg_dev, cfg_user)
