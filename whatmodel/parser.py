"""Layer 1: read a model name without touching the network.

The *structure* of a name (host, namespace, model, tag, file) is syntax, so it
is handled here. What each token *means* comes only from ``tokens.yaml``:

* a token that matches no entry is reported as unknown, never guessed;
* a token that matches several entries is reported as ambiguous, with every
  reading kept.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from whatmodel.dictionary import (
    Dictionary,
    Piece,
    Reading,
    get_dictionary,
    normalize,
    split_pieces,
)

#: The tag Ollama uses when a name has none.
DEFAULT_TAG = "latest"

#: Hosts that mean "this is a Hugging Face repo". Used for routing, not meaning.
HUGGINGFACE_HOSTS = frozenset({"hf.co", "huggingface.co"})

MAX_NAME_LENGTH = 512

KIND_OLLAMA = "ollama"
KIND_HUGGINGFACE = "huggingface"
KIND_FILE = "file"

_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_URL_TAIL = re.compile(r"[?#]")


@dataclass(frozen=True)
class Token:
    """One piece of the name and what the dictionary says about it."""

    text: str
    #: Where in the name it sits: "host", "org", "model" or "tag".
    part: str
    #: No readings means unknown, one means known, several mean ambiguous.
    readings: tuple[Reading, ...] = ()

    @property
    def status(self) -> str:
        if not self.readings:
            return "unknown"
        return "known" if len(self.readings) == 1 else "ambiguous"

    @property
    def reading(self) -> Reading | None:
        """The single reading of a known token, or ``None`` if unknown or ambiguous."""
        return self.readings[0] if len(self.readings) == 1 else None

    @property
    def category(self) -> str | None:
        reading = self.reading
        return reading.category if reading is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "part": self.part,
            "status": self.status,
            "source": "name",
            "readings": [reading.to_dict() for reading in self.readings],
        }


@dataclass(frozen=True)
class Implied:
    """Something the name does not spell out but still implies, such as the default tag."""

    reading: Reading
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.reading.to_dict(), "reason": self.reason, "source": "name"}


@dataclass(frozen=True)
class ParsedName:
    """Everything Layer 1 can tell from the name alone."""

    raw: str
    #: How the name can be read: "ollama", "huggingface" or "file". Two entries
    #: mean the name fits both and nothing in it settles which.
    kinds: tuple[str, ...]
    host: str | None
    namespace: str | None
    model: str
    tag: str | None
    tag_is_default: bool
    filename: str | None
    tokens: tuple[Token, ...]
    implied: tuple[Implied, ...] = ()

    def values(self, category: str) -> tuple[str, ...]:
        """Values in ``category``: unambiguous tokens in name order, then anything implied.

        Ambiguous tokens are left out, because the name does not settle what they mean.
        """
        explicit = tuple(
            token.reading.value
            for token in self.tokens
            if token.reading is not None and token.reading.category == category
        )
        implied = tuple(
            item.reading.value for item in self.implied if item.reading.category == category
        )
        return explicit + implied

    def _first(self, category: str) -> str | None:
        values = self.values(category)
        return values[0] if values else None

    @property
    def org(self) -> str | None:
        return self._first("org")

    @property
    def family(self) -> str | None:
        return self._first("family")

    @property
    def version(self) -> str | None:
        return self._first("version")

    @property
    def size(self) -> str | None:
        return self._first("size")

    @property
    def effective_size(self) -> str | None:
        return self._first("effective_size")

    @property
    def active_params(self) -> str | None:
        return self._first("active_params")

    @property
    def tier(self) -> str | None:
        return self._first("tier")

    @property
    def tuning(self) -> str | None:
        return self._first("tuning")

    @property
    def quantization(self) -> str | None:
        return self._first("quantization")

    @property
    def format(self) -> str | None:
        return self._first("format")

    @property
    def context(self) -> str | None:
        return self._first("context")

    @property
    def build(self) -> str | None:
        return self._first("build")

    @property
    def hosting(self) -> str | None:
        return self._first("hosting")

    @property
    def architecture(self) -> str | None:
        return self._first("architecture")

    @property
    def unknown(self) -> tuple[str, ...]:
        """The text of every token the dictionary does not know."""
        return tuple(token.text for token in self.tokens if not token.readings)

    @property
    def ambiguous(self) -> tuple[Token, ...]:
        """Tokens with more than one reading."""
        return tuple(token for token in self.tokens if len(token.readings) > 1)

    @property
    def is_ambiguous(self) -> bool:
        """True when the name's format or any of its tokens has more than one reading."""
        return len(self.kinds) > 1 or bool(self.ambiguous)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.raw,
            "kinds": list(self.kinds),
            "parts": {
                "host": self.host,
                "namespace": self.namespace,
                "model": self.model,
                "tag": self.tag,
                "tag_is_default": self.tag_is_default,
                "filename": self.filename,
            },
            "tokens": [token.to_dict() for token in self.tokens],
            "implied": [item.to_dict() for item in self.implied],
        }


@dataclass(frozen=True)
class _Structure:
    kinds: tuple[str, ...]
    host: str | None
    namespace: str | None
    model: str
    tag: str | None
    tag_is_default: bool
    filename: str | None
    #: The text that is split into tokens for the "model" part.
    token_text: str


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    readings: tuple[Reading, ...]


def _is_file(segment: str, dictionary: Dictionary) -> bool:
    """True when ``segment`` ends in an extension the dictionary lists as a format."""
    stem, dot, extension = segment.rpartition(".")
    if not dot or not stem or ":" in segment:
        return False
    return bool(dictionary.match(normalize(extension), category="format"))


def _split_structure(raw: str, dictionary: Dictionary) -> _Structure:
    text = raw.replace("\\", "/")
    body = _SCHEME.sub("", text)
    has_scheme = body != text
    if has_scheme:
        body = _URL_TAIL.split(body, maxsplit=1)[0]
    segments = [segment for segment in body.split("/") if segment]
    if not segments:
        raise ValueError(f"no model name in {raw!r}")

    last = segments[-1]
    if _is_file(last, dictionary):
        stem = last.rpartition(".")[0]
        return _Structure((KIND_FILE,), None, None, stem, None, False, last, last)

    # A pasted Hugging Face URL may carry a path after org/repo; drop it.
    if len(segments) > 3 and segments[0].lower() in HUGGINGFACE_HOSTS:
        segments = segments[:3]
        last = segments[-1]

    model = last.split("@", 1)[0]  # an @digest pins a build; it carries no meaning
    tag: str | None = None
    if ":" in model:
        model, _, found = model.rpartition(":")
        tag = found or None
    if not model:
        raise ValueError(f"no model name in {raw!r}")

    lead = segments[:-1]
    host: str | None = None
    if has_scheme and lead:
        host, lead = lead[0], lead[1:]
    elif len(lead) >= 2:
        host, lead = "/".join(lead[:-1]), lead[-1:]
    namespace = lead[-1] if lead else None

    if host is not None:
        if host.lower() in HUGGINGFACE_HOSTS and tag is None:
            kinds = (KIND_HUGGINGFACE,) if has_scheme else (KIND_HUGGINGFACE, KIND_OLLAMA)
        else:
            kinds = (KIND_OLLAMA,)
    elif namespace is not None and tag is None:
        # "a/b" fits both a Hugging Face repo and an Ollama namespace/model.
        kinds = (KIND_HUGGINGFACE, KIND_OLLAMA)
    else:
        kinds = (KIND_OLLAMA,)

    tag_is_default = False
    if tag is None and KIND_OLLAMA in kinds:
        tag, tag_is_default = DEFAULT_TAG, True
    return _Structure(kinds, host, namespace, model, tag, tag_is_default, None, model)


def _positional(text: str, part: str, category: str, dictionary: Dictionary) -> Token:
    """Build the token for a part whose role is fixed by its position in the name."""
    readings = dictionary.match(normalize(text), category=category, ignore_context=True)
    if not readings:
        # The position says what it is even though the dictionary has no entry.
        description = dictionary.categories[category].description
        readings = (Reading(category, label=text, explanation=description, value=text.lower()),)
    return Token(text=text, part=part, readings=readings)


def _raw_offset(run: Sequence[Piece], cut: int) -> int | None:
    """Translate an offset in a normalized run back to an offset in the source text."""
    remaining = cut
    for piece in run:
        size = len(piece.key)
        if size != len(piece.text):
            return None  # lowercasing changed the length; do not risk a wrong cut
        if remaining < size:
            return piece.start + remaining
        remaining -= size + 1
    return None


def _tokenize(text: str, part: str, dictionary: Dictionary) -> list[Token]:
    """Split ``text`` into tokens, choosing the reading with the fewest unknown parts.

    Adjacent parts are tried together (longest run first) so that multi-part
    tokens such as ``Q4_K_M`` or ``gpt-oss`` survive. Ties go to fewer tokens.
    """
    pieces = split_pieces(text)
    keys = [piece.key for piece in pieces]
    count = len(pieces)

    @cache
    def best(index: int, prev: frozenset[str]) -> tuple[int, int, tuple[_Span, ...]]:
        if index == count:
            return 0, 0, ()
        options: list[tuple[int, int, tuple[_Span, ...]]] = []
        for length in range(min(dictionary.max_parts, count - index), 0, -1):
            run = pieces[index : index + length]
            candidate = "_".join(keys[index : index + length])
            start, end = run[0].start, run[-1].end

            readings = dictionary.match(candidate, prev)
            if readings:
                after = frozenset(reading.category for reading in readings)
                unknown, used, rest = best(index + length, after)
                options.append((unknown, used + 1, (_Span(start, end, readings), *rest)))

            fused = dictionary.split_fused(candidate)
            cut = _raw_offset(run, fused.cut) if fused is not None else None
            if fused is not None and cut is not None:
                after = frozenset(reading.category for reading in fused.versions)
                unknown, used, rest = best(index + length, after)
                spans = (_Span(start, cut, fused.families), _Span(cut, end, fused.versions))
                options.append((unknown, used + 2, (*spans, *rest)))

        unknown, used, rest = best(index + 1, frozenset())
        piece = pieces[index]
        options.append((unknown + 1, used + 1, (_Span(piece.start, piece.end, ()), *rest)))
        return min(options, key=lambda option: (option[0], option[1]))

    tokens: list[Token] = []
    pending: _Span | None = None  # a run of adjacent unknown parts, merged into one token
    for span in best(0, frozenset())[2]:
        if not span.readings:
            pending = span if pending is None else _Span(pending.start, span.end, ())
            continue
        if pending is not None:
            tokens.append(Token(text[pending.start : pending.end], part))
            pending = None
        tokens.append(Token(text[span.start : span.end], part, span.readings))
    if pending is not None:
        tokens.append(Token(text[pending.start : pending.end], part))
    return tokens


def _has_rival(token: Token, tokens: Sequence[Token]) -> bool:
    """True when the name holds a different token of the same category.

    A name such as "Qwen3.5-9B-Claude-4.6-Opus-Distilled" mentions two families.
    Nothing says which one the model is, so neither is allowed to imply anything.
    """
    reading = token.reading
    if reading is None:
        return False
    return any(
        other.reading is not None
        and other.reading.category == reading.category
        and other.reading.value != reading.value
        for other in tokens
    )


def _implied(
    structure: _Structure, tokens: Sequence[Token], dictionary: Dictionary
) -> list[Implied]:
    """Collect what the name implies without spelling it out."""
    explicit = {reading.category for token in tokens for reading in token.readings}
    implied: list[Implied] = []

    # 1. Tokens whose dictionary entry says they also settle another category.
    #    Unknown and ambiguous tokens imply nothing, and neither does a token that
    #    shares its category with a different one (see _has_rival).
    for token in tokens:
        reading = token.reading
        if reading is None or _has_rival(token, tokens):
            continue
        for category, target in reading.implies:
            found = dictionary.describe(target, category=category)
            if found is None or category in explicit:
                continue
            if all(item.reading != found for item in implied):
                implied.append(Implied(found, f"implied by '{token.text}' in the name"))

    # 2. The tag Ollama fills in when the name has none.
    if structure.tag_is_default and structure.kinds == (KIND_OLLAMA,):
        implied.extend(
            Implied(reading, "the name has no tag, so Ollama uses its default")
            for reading in dictionary.match(normalize(DEFAULT_TAG), ignore_context=True)
        )

    # 3. Category defaults, for categories nothing above has spoken for.
    covered = explicit | {item.reading.category for item in implied}
    for category in dictionary.categories.values():
        if category.default is not None and category.id not in covered:
            reading = Reading(
                category.id,
                label=category.default.label,
                explanation=category.default.explanation,
                value=category.default.value,
            )
            implied.append(Implied(reading, f"the name has no {category.id} token"))
    return implied


def parse(name: str) -> ParsedName:
    """Explain a model name using only the name itself (no network).

    Accepts Ollama names (``[host/][namespace/]model[:tag]``), Hugging Face
    repos (``org/repo``) and GGUF-style file names. Raises ``ValueError`` only
    when there is no name to read at all.
    """
    raw = name.strip()
    if not raw:
        raise ValueError("model name is empty")
    if len(raw) > MAX_NAME_LENGTH:
        raise ValueError(f"model name is longer than {MAX_NAME_LENGTH} characters")

    dictionary = get_dictionary()
    structure = _split_structure(raw, dictionary)

    tokens: list[Token] = []
    if structure.host is not None:
        tokens.append(_positional(structure.host, "host", "host", dictionary))
    if structure.namespace is not None:
        tokens.append(_positional(structure.namespace, "org", "org", dictionary))
    tokens.extend(_tokenize(structure.token_text, "model", dictionary))
    if structure.tag is not None and not structure.tag_is_default:
        tokens.extend(_tokenize(structure.tag, "tag", dictionary))

    return ParsedName(
        raw=raw,
        kinds=structure.kinds,
        host=structure.host,
        namespace=structure.namespace,
        model=structure.model,
        tag=structure.tag,
        tag_is_default=structure.tag_is_default,
        filename=structure.filename,
        tokens=tuple(tokens),
        implied=tuple(_implied(structure, tokens, dictionary)),
    )
