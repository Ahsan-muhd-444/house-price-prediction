"""Exception types raised deliberately by amesml.

Every error derives from :class:`AmesMLError`, so callers can catch the whole
family in one place. Each also derives from the closest built-in exception, so
code that already handles ``ValueError`` or ``RuntimeError`` keeps working.
"""

from __future__ import annotations


class AmesMLError(Exception):
    """Base class for all errors raised by amesml."""


class AmesSchemaError(AmesMLError, ValueError):
    """Data does not match the columns or types the pipeline expects."""


class ConfigError(AmesMLError, ValueError):
    """A training configuration is invalid."""


class DatasetDownloadError(AmesMLError, RuntimeError):
    """The dataset could not be downloaded, or it failed its integrity check."""


class ModelLoadError(AmesMLError, RuntimeError):
    """A saved model is missing, incomplete, or does not match its metadata."""
