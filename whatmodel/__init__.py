"""whatmodel: explain LLM model names in plain English.

Two layers:

* :func:`parse` reads the name itself, offline.
* :func:`explain` does the same and then looks up the facts a name cannot show.

Also here: :func:`explain_many` for a list of names, :func:`compare` to line
names up field by field, and :func:`meanings` to ask what one token means.

    >>> import whatmodel
    >>> print(whatmodel.parse("qwen3.5:4b"))
    Qwen 3.5 (Alibaba); 4B parameters
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from whatmodel.compare import ComparedField, Comparison, compare
from whatmodel.dictionary import Reading, get_dictionary, normalize
from whatmodel.lookup import Fact, ModelInfo, explain, explain_many
from whatmodel.parser import Implied, ParsedName, Token, parse


def meanings(token: str) -> tuple[Reading, ...]:
    """Every meaning the dictionary has for one token, such as ``"qat"`` or ``"Q4_K_M"``.

    Returns an empty tuple for a token it does not know, and more than one
    reading when the token is ambiguous. Nothing is guessed.
    """
    return get_dictionary().match(normalize(token), ignore_context=True)


try:
    __version__ = _distribution_version("whatmodel")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "ComparedField",
    "Comparison",
    "Fact",
    "Implied",
    "ModelInfo",
    "ParsedName",
    "Reading",
    "Token",
    "__version__",
    "compare",
    "explain",
    "explain_many",
    "meanings",
    "parse",
]
