from .default_config import _C
from third_party.config import CfgNode as CN
from .constants import PROJECT_NAME


def get_cfg_defaults() -> CN:
    """
    Return a fresh clone of the default config.
    """
    return _C.clone()


__all__ = ["get_cfg_defaults", "PROJECT_NAME"]
