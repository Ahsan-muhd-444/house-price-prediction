"""amesml: reproducible Ames Housing price prediction as a package, CLI and API."""

from __future__ import annotations

import logging

from ._version import __version__

# Library code only logs; applications decide where the messages go.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__"]
