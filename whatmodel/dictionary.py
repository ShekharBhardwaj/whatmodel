"""The token dictionary: loads ``tokens.yaml`` and answers "what does this token mean?".

This module knows the file's schema and how matching works. It knows nothing
about any particular token: every meaning comes from the data file.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

#: Categories the parser refers to by id. ``tokens.yaml`` may define more.
CORE_CATEGORIES = ("host", "org", "family", "version", "format")

#: Longest run of name parts that is ever tried as one token.
MAX_PARTS_LIMIT = 6

_TOP_LEVEL_KEYS = frozenset({"schema", "categories", "tokens"})
_CATEGORY_KEYS = frozenset({"heading", "description", "default"})
_DEFAULT_KEYS = frozenset({"label", "value", "explanation"})
_ENTRY_KEYS = frozenset(
    {
        "pattern",
        "regex",
        "category",
        "label",
        "explanation",
        "value",
        "examples",
        "after",
        "maker",
        "implies",
    }
)
_REQUIRED_ENTRY_KEYS = ("pattern", "category", "label", "explanation")

# A name part is a run of characters with no separator in it. "-", "_", "." and
# whitespace separate parts, except that a dot between two digits is a decimal
# point (so "3.5" and "v0.3" stay whole).
_PIECE = re.compile(r"(?:[^\s\-_.]|(?<=\d)\.(?=\d))+")


class DictionaryError(ValueError):
    """Raised when ``tokens.yaml`` does not follow the schema."""


@dataclass(frozen=True)
class Piece:
    """One separator-free part of a name, with its position in the source text."""

    text: str
    start: int
    end: int

    @property
    def key(self) -> str:
        """The part as it is matched: lowercased."""
        return self.text.lower()


def split_pieces(text: str) -> list[Piece]:
    """Split ``text`` on separators, keeping decimal points inside numbers."""
    return [Piece(match.group(), match.start(), match.end()) for match in _PIECE.finditer(text)]


def normalize(text: str) -> str:
    """Return the form tokens are matched in: lowercase, parts joined by ``_``."""
    return "_".join(piece.key for piece in split_pieces(text))


class _TemplateFormatter(string.Formatter):
    """``str.format`` with one extra conversion: ``!u`` uppercases the value."""

    def convert_field(self, value: Any, conversion: str | None) -> Any:
        if conversion == "u":
            return str(value).upper()
        return super().convert_field(value, conversion)


_FORMATTER = _TemplateFormatter()


def _render(template: str, fields: Mapping[str, str]) -> str:
    return _FORMATTER.vformat(template, (), fields)


def _template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in _FORMATTER.parse(template) if name is not None}


@dataclass(frozen=True)
class CategoryDefault:
    """What a category shows when no token in the name belongs to it."""

    label: str
    value: str
    explanation: str


@dataclass(frozen=True)
class Category:
    """A kind of token, and how it is displayed."""

    id: str
    heading: str
    description: str
    default: CategoryDefault | None = None


@dataclass(frozen=True)
class Reading:
    """One meaning of a token, as given by one dictionary entry."""

    category: str
    label: str
    explanation: str
    value: str
    maker: str | None = None
    #: (category, token) pairs this reading also implies; see ``implies`` in tokens.yaml.
    implies: tuple[tuple[str, str], ...] = field(default=(), repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category,
            "value": self.value,
            "label": self.label,
            "explanation": self.explanation,
        }
        if self.maker is not None:
            data["maker"] = self.maker
        return data


@dataclass(frozen=True, eq=False)
class Entry:
    """One entry of ``tokens.yaml``, compiled for matching."""

    index: int
    patterns: tuple[str, ...]
    is_regex: bool
    category: str
    label: str
    explanation: str
    value: str | None = None
    examples: tuple[str, ...] = ()
    after: str | None = None
    maker: str | None = None
    implies: tuple[tuple[str, str], ...] = ()
    #: Literal entries: normalized literal -> the literal as written, lowercased.
    literals: Mapping[str, str] = field(default_factory=dict, repr=False)
    #: Regex entries: one compiled pattern per ``patterns`` item.
    compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False)

    def groups(self, candidate: str) -> dict[str, str] | None:
        """Return the named groups if ``candidate`` matches this entry, else ``None``."""
        if not self.is_regex:
            return {} if candidate in self.literals else None
        for pattern in self.compiled:
            match = pattern.fullmatch(candidate)
            if match is not None:
                return {name: value or "" for name, value in match.groupdict().items()}
        return None

    def read(self, candidate: str, groups: Mapping[str, str]) -> Reading:
        """Build the reading this entry gives for a matching ``candidate``."""
        fields = {"text": candidate, **groups}
        if self.value is not None:
            value = _render(self.value, fields)
        elif self.is_regex:
            value = candidate
        else:
            value = self.literals[candidate]
        return Reading(
            category=self.category,
            label=_render(self.label, fields),
            explanation=_render(self.explanation, fields),
            value=value,
            maker=self.maker,
            implies=self.implies,
        )


@dataclass(frozen=True)
class FusedSplit:
    """A part such as ``qwen3.5`` read as a known family followed by a version."""

    cut: int
    families: tuple[Reading, ...]
    versions: tuple[Reading, ...]


class Dictionary:
    """All categories and entries, indexed for lookup."""

    def __init__(self, categories: dict[str, Category], entries: tuple[Entry, ...]) -> None:
        self.categories = categories
        self.entries = entries
        self._literal_index: dict[str, list[Entry]] = {}
        self._regex_entries: list[Entry] = []
        family_literals: set[str] = set()
        max_parts = 1
        for entry in entries:
            if entry.is_regex:
                self._regex_entries.append(entry)
                max_parts = max(max_parts, *(p.count("_") + 1 for p in entry.patterns))
                continue
            for literal in entry.literals:
                self._literal_index.setdefault(literal, []).append(entry)
                max_parts = max(max_parts, literal.count("_") + 1)
                if entry.category == "family":
                    family_literals.add(literal)
        self.max_parts = min(max_parts, MAX_PARTS_LIMIT)
        self._family_literals = sorted(family_literals, key=len, reverse=True)

    def match(
        self,
        candidate: str,
        prev: frozenset[str] = frozenset(),
        *,
        category: str | None = None,
        ignore_context: bool = False,
    ) -> tuple[Reading, ...]:
        """Return every reading of ``candidate`` (a normalized token), in file order.

        ``prev`` holds the categories of the token just before it, which is what
        ``after`` constraints are checked against. ``category`` restricts the
        search to one category. ``ignore_context`` skips ``after`` constraints.
        """
        hits: list[tuple[int, Reading]] = []
        for entry in (*self._literal_index.get(candidate, ()), *self._regex_entries):
            if category is not None and entry.category != category:
                continue
            if not ignore_context and entry.after is not None and entry.after not in prev:
                continue
            groups = entry.groups(candidate)
            if groups is not None:
                hits.append((entry.index, entry.read(candidate, groups)))
        hits.sort(key=lambda hit: hit[0])
        readings: list[Reading] = []
        for _, reading in hits:
            if reading not in readings:
                readings.append(reading)
        return tuple(readings)

    def split_fused(self, candidate: str) -> FusedSplit | None:
        """Read ``candidate`` as ``<known family><version>`` with nothing between them."""
        for literal in self._family_literals:
            if len(candidate) <= len(literal) or not candidate.startswith(literal):
                continue
            rest = candidate[len(literal) :]
            if rest.startswith("_"):
                continue
            versions = self.match(rest, frozenset({"family"}), category="version")
            if versions:
                families = self.match(literal, category="family", ignore_context=True)
                return FusedSplit(cut=len(literal), families=families, versions=versions)
        return None

    def describe(self, text: str, *, category: str | None = None) -> Reading | None:
        """Explain a free-standing value, such as a quantization level from an API.

        Returns ``None`` unless exactly one entry matches, so nothing is guessed.
        """
        readings = self.match(normalize(text), category=category, ignore_context=True)
        return readings[0] if len(readings) == 1 else None

    def entries_in(self, category: str) -> tuple[Entry, ...]:
        """Return the entries of one category, in file order."""
        return tuple(entry for entry in self.entries if entry.category == category)


def _fail(where: str, message: str) -> DictionaryError:
    return DictionaryError(f"{where}: {message}")


def _require_str(where: str, data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"'{key}' must be a non-empty string (quote numbers such as '3')")
    return value.strip()


def _optional_str(where: str, data: Mapping[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    return _require_str(where, data, key)


def _str_list(where: str, value: object, key: str) -> tuple[str, ...]:
    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(item, str) and item.strip() for item in items):
        raise _fail(
            where,
            f"'{key}' must be a string or a list of strings (quote numbers such as '2507')",
        )
    return tuple(item.strip() for item in items)


def _build_categories(raw: object) -> dict[str, Category]:
    if not isinstance(raw, dict) or not raw:
        raise DictionaryError("'categories' must be a non-empty mapping")
    categories: dict[str, Category] = {}
    for category_id, body in raw.items():
        where = f"categories.{category_id}"
        if not isinstance(category_id, str) or not isinstance(body, dict):
            raise _fail(where, "must be a mapping with 'heading' and 'description'")
        unknown = set(body) - _CATEGORY_KEYS
        if unknown:
            raise _fail(where, f"unknown key(s): {', '.join(sorted(map(str, unknown)))}")
        categories[category_id] = Category(
            id=category_id,
            heading=_require_str(where, body, "heading"),
            description=_require_str(where, body, "description"),
            default=_build_default(where, body),
        )
    missing = [name for name in CORE_CATEGORIES if name not in categories]
    if missing:
        raise DictionaryError(f"'categories' is missing required id(s): {', '.join(missing)}")
    return categories


def _build_default(where: str, body: Mapping[str, Any]) -> CategoryDefault | None:
    if "default" not in body:
        return None
    raw = body["default"]
    if isinstance(raw, str) and raw.strip():
        label = raw.strip()
        return CategoryDefault(label, value=label, explanation=str(body["description"]).strip())
    if isinstance(raw, dict) and not set(raw) - _DEFAULT_KEYS:
        inner = f"{where}.default"
        label = _require_str(inner, raw, "label")
        return CategoryDefault(
            label,
            value=_optional_str(inner, raw, "value") or label,
            explanation=_require_str(inner, raw, "explanation"),
        )
    raise _fail(
        where,
        "'default' must be a string, or a mapping with 'label', 'explanation' and optional 'value'",
    )


def _build_implies(
    where: str, raw: object, category: str, categories: Mapping[str, Category]
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict) or not raw:
        raise _fail(where, "'implies' must map a category id to a token, as in {hosting: cloud}")
    pairs: list[tuple[str, str]] = []
    for target, token in raw.items():
        if target not in categories:
            raise _fail(where, f"'implies' names unknown category '{target}'")
        if target == category:
            raise _fail(where, "'implies' cannot point at the entry's own category")
        if not isinstance(token, str) or not token.strip():
            raise _fail(where, f"'implies.{target}' must be a token defined in that category")
        pairs.append((target, token.strip()))
    return tuple(pairs)


def _check_templates(
    where: str, templates: Mapping[str, str | None], allowed: frozenset[str]
) -> None:
    for key, template in templates.items():
        if template is None:
            continue
        try:
            used = _template_fields(template)
        except ValueError as exc:
            raise _fail(where, f"'{key}' is not a valid template: {exc}") from exc
        unknown = used - allowed
        if unknown:
            raise _fail(
                where,
                f"'{key}' uses {{{', '.join(sorted(unknown))}}}, but only "
                f"{{{', '.join(sorted(allowed))}}} is available",
            )


def _build_entry(index: int, raw: object, categories: Mapping[str, Category]) -> Entry:
    where = f"tokens[{index}]"
    if not isinstance(raw, dict):
        raise _fail(where, "must be a mapping")
    if "pattern" in raw:
        where = f"tokens[{index}] (pattern {raw['pattern']!r})"
    unknown = set(raw) - _ENTRY_KEYS
    if unknown:
        raise _fail(where, f"unknown key(s): {', '.join(sorted(map(str, unknown)))}")
    for key in _REQUIRED_ENTRY_KEYS:
        if key not in raw:
            raise _fail(where, f"missing required key '{key}'")

    patterns = _str_list(where, raw["pattern"], "pattern")
    category = _require_str(where, raw, "category")
    if category not in categories:
        raise _fail(where, f"unknown category '{category}'")
    is_regex = raw.get("regex", False)
    if not isinstance(is_regex, bool):
        raise _fail(where, "'regex' must be true or false")
    after = _optional_str(where, raw, "after")
    if after is not None and after not in categories:
        raise _fail(where, f"'after' names unknown category '{after}'")
    maker = _optional_str(where, raw, "maker")
    if maker is not None and category != "family":
        raise _fail(where, "'maker' is only allowed on family entries")
    examples = _str_list(where, raw["examples"], "examples") if "examples" in raw else ()
    implies = (
        _build_implies(where, raw["implies"], category, categories) if "implies" in raw else ()
    )
    label = _require_str(where, raw, "label")
    explanation = _require_str(where, raw, "explanation")
    value = _optional_str(where, raw, "value")

    literals: dict[str, str] = {}
    compiled: tuple[re.Pattern[str], ...] = ()
    allowed = frozenset({"text"})
    if is_regex:
        if not examples:
            raise _fail(where, "regex entries need 'examples' (tokens the pattern must match)")
        try:
            compiled = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        except re.error as exc:
            raise _fail(where, f"invalid regular expression: {exc}") from exc
        group_sets = [set(pattern.groupindex) for pattern in compiled]
        if any("text" in groups for groups in group_sets):
            raise _fail(where, "a regex group cannot be named 'text'")
        allowed = allowed | frozenset(set.intersection(*group_sets))
    else:
        for pattern in patterns:
            key = normalize(pattern)
            if not key:
                raise _fail(where, f"literal {pattern!r} has nothing to match")
            literals[key] = pattern.lower()
    _check_templates(where, {"label": label, "explanation": explanation, "value": value}, allowed)

    entry = Entry(
        index=index,
        patterns=patterns,
        is_regex=is_regex,
        category=category,
        label=label,
        explanation=explanation,
        value=value,
        examples=examples,
        after=after,
        maker=maker,
        implies=implies,
        literals=literals,
        compiled=compiled,
    )
    for example in examples:
        if entry.groups(normalize(example)) is None:
            raise _fail(where, f"example {example!r} does not match the pattern")
    return entry


def build_dictionary(data: object) -> Dictionary:
    """Validate parsed YAML and build a :class:`Dictionary` from it."""
    if not isinstance(data, dict):
        raise DictionaryError("the file must be a mapping with 'schema', 'categories', 'tokens'")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise DictionaryError(f"unknown top-level key(s): {', '.join(sorted(map(str, unknown)))}")
    if data.get("schema") != SCHEMA_VERSION:
        raise DictionaryError(f"'schema' must be {SCHEMA_VERSION}, found {data.get('schema')!r}")
    categories = _build_categories(data.get("categories"))
    raw_tokens = data.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise DictionaryError("'tokens' must be a non-empty list")
    entries = tuple(_build_entry(index, raw, categories) for index, raw in enumerate(raw_tokens))

    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        for literal in entry.literals:
            key = (entry.category, literal)
            if key in seen:
                raise DictionaryError(
                    f"tokens[{entry.index}]: literal {literal!r} is already defined for "
                    f"category '{entry.category}' in tokens[{seen[key]}]"
                )
            seen[key] = entry.index

    dictionary = Dictionary(categories, entries)
    for entry in entries:
        for target, token in entry.implies:
            if dictionary.describe(token, category=target) is None:
                raise DictionaryError(
                    f"tokens[{entry.index}]: 'implies.{target}: {token}' must match exactly one "
                    f"entry in category '{target}'"
                )
    return dictionary


def load_dictionary(path: str | Path | None = None) -> Dictionary:
    """Load a dictionary file. With no ``path``, load the one shipped in the package."""
    if path is None:
        origin = "tokens.yaml"
        text = resources.files("whatmodel").joinpath("tokens.yaml").read_text(encoding="utf-8")
    else:
        origin = str(path)
        text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DictionaryError(f"{origin}: not valid YAML: {exc}") from exc
    try:
        return build_dictionary(data)
    except DictionaryError as exc:
        raise DictionaryError(f"{origin}: {exc}") from exc


@lru_cache(maxsize=1)
def get_dictionary() -> Dictionary:
    """Return the shipped dictionary, loaded once per process."""
    return load_dictionary()
