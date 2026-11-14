"""Read the metadata at the start of a GGUF file.

A GGUF file opens with a header of key/value pairs that describe the model:
its architecture, context length, how it was compressed, and more. Only that
header is read here. The weights that follow are never touched, so this stays
fast on files of any size.

The layout follows the GGUF specification (docs/gguf.md in ggml-org/ggml) and
the file-type list follows ``enum llama_ftype`` in llama.cpp's llama.h, both
checked in November 2026.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

#: Scalar value types: type id -> struct format character.
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_STRING, _ARRAY = 8, 9

#: Guards against a corrupt or hostile file. Real files stay far below these.
MAX_PAIRS = 100_000
MAX_KEY_BYTES = 65_535
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_ARRAY_ITEMS = 50_000_000
#: Arrays longer than this are skipped, not returned. This drops the vocabulary.
KEEP_ARRAY_ITEMS = 64

#: ``general.file_type`` values, from ``enum llama_ftype`` in llama.h.
FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
    42: "F8_E4M3",
}


class GGUFError(ValueError):
    """The file is not a GGUF file, or its header cannot be read."""


class _Reader:
    def __init__(self, stream: BinaryIO, order: str) -> None:
        self.stream = stream
        self.order = order

    def take(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise GGUFError("the file ends in the middle of its header")
        return data

    def number(self, code: str) -> Any:
        return struct.unpack(self.order + code, self.take(struct.calcsize(code)))[0]

    def skip(self, size: int) -> None:
        self.stream.seek(size, 1)

    def string(self, limit: int = MAX_STRING_BYTES, *, keep: bool = True) -> str | None:
        size = self.number("Q")
        if size > limit:
            raise GGUFError("a text value in the header is implausibly long")
        if not keep:
            self.skip(size)
            return None
        return self.take(size).decode("utf-8", errors="replace")

    def value(self, kind: int, *, keep: bool = True) -> Any:
        if kind in _SCALARS:
            return self.number(_SCALARS[kind])
        if kind == _STRING:
            return self.string(keep=keep)
        if kind == _ARRAY:
            inner, count = self.number("I"), self.number("Q")
            if count > MAX_ARRAY_ITEMS or inner == _ARRAY:
                raise GGUFError("a list in the header is implausibly large")
            wanted = keep and count <= KEEP_ARRAY_ITEMS
            if not wanted and inner in _SCALARS:
                self.skip(count * struct.calcsize(_SCALARS[inner]))
                return None
            items = [self.value(inner, keep=wanted) for _ in range(count)]
            return items if wanted else None
        raise GGUFError(f"unknown value type {kind} in the header")


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Return the header's key/value pairs. Long lists, such as the vocabulary, are left out.

    Raises :class:`GGUFError` if the file is not GGUF or its header is damaged,
    and ``OSError`` if it cannot be opened.
    """
    with Path(path).open("rb") as stream:
        if stream.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        raw_version = stream.read(4)
        if len(raw_version) != 4:
            raise GGUFError("the file ends in the middle of its header")
        # The version is a small number, so a huge one means the other byte order.
        order = "<" if struct.unpack("<I", raw_version)[0] < 65_536 else ">"
        version = struct.unpack(order + "I", raw_version)[0]
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} is not supported")
        reader = _Reader(stream, order)
        reader.number("Q")  # tensor count
        pairs = reader.number("Q")
        if pairs > MAX_PAIRS:
            raise GGUFError("the header claims an implausible number of entries")
        metadata: dict[str, Any] = {}
        try:
            for _ in range(pairs):
                key = reader.string(MAX_KEY_BYTES)
                value = reader.value(reader.number("I"))
                if key is not None and value is not None:
                    metadata[key] = value
        except (struct.error, OSError, OverflowError) as exc:
            raise GGUFError("the header is damaged") from exc
        return metadata


def file_type_name(metadata: dict[str, Any]) -> str | None:
    """The compression named by ``general.file_type``, such as ``Q4_K_M``."""
    value = metadata.get("general.file_type")
    return FILE_TYPES.get(value) if isinstance(value, int) else None


__all__ = ["FILE_TYPES", "GGUFError", "file_type_name", "read_metadata"]
