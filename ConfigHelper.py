from pathlib import Path
from types import SimpleNamespace


def load_config() -> SimpleNamespace:
    """
    Loads config from `config.default.py` and overrides it with the config from `config.py` if it exists.

    :return: the combined configuration
    """

    from config_default import config as cfg_default
    if Path("config.py").exists():
        from config import config as cfg_user
    else:
        cfg_user = {}

    return SimpleNamespace(**(cfg_default | cfg_user))
