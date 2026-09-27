"""Transitional module redirecting kea_cyberbully imports to cyberbully."""

import warnings

warnings.warn(
    "The 'kea-cyberbully' package is deprecated and will receive no further updates. "
    "Please install and use 'cyberbully' instead: pip install cyberbully",
    DeprecationWarning,
    stacklevel=2,
)

from cyberbully import *  # noqa: F401, F403
