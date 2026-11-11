"""whatmodel: explain LLM model names in plain English.

Two layers:

* :func:`parse` reads the name itself, offline.
* :func:`explain` does the same and then looks up the facts a name cannot show.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from whatmodel.dictionary import Reading
from whatmodel.lookup import Fact, ModelInfo, explain
from whatmodel.parser import Implied, ParsedName, Token, parse

try:
    __version__ = _distribution_version("whatmodel")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Fact",
    "Implied",
    "ModelInfo",
    "ParsedName",
    "Reading",
    "Token",
    "__version__",
    "explain",
    "parse",
]
