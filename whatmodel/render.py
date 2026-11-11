"""Text output. Layout lives here; every word about a token comes from tokens.yaml.

Facts from the name and facts from lookups are kept in two separate blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from whatmodel.dictionary import Dictionary, Reading, get_dictionary
from whatmodel.lookup import (
    SOURCE_HUGGINGFACE,
    SOURCE_OLLAMA,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_UNREACHABLE,
    Fact,
    ModelInfo,
    format_count,
)
from whatmodel.parser import KIND_FILE, KIND_HUGGINGFACE, KIND_OLLAMA, ParsedName

LOOKUP_HEADING = "Not in the name"

#: Row order and headings for looked-up facts.
FACT_HEADINGS = {
    "hosting": "Runs",
    "parameter_count": "Parameters",
    "quantization": "Compression",
    "architecture": "Architecture",
    "context_length": "Context",
    "tools": "Tools",
    "vision": "Vision",
    "thinking": "Thinking",
    "license": "License",
    "file_size": "File size",
}

_KIND_PHRASES = {
    KIND_OLLAMA: "an Ollama model",
    KIND_HUGGINGFACE: "a Hugging Face repo",
    KIND_FILE: "a model file",
}

_SOURCE_NAMES = {SOURCE_OLLAMA: "local Ollama", SOURCE_HUGGINGFACE: "Hugging Face"}

_MISS_REASONS = {
    (SOURCE_OLLAMA, STATUS_NOT_FOUND): "not installed in local Ollama",
    (SOURCE_OLLAMA, STATUS_UNREACHABLE): "local Ollama is not running",
    (SOURCE_OLLAMA, STATUS_TIMEOUT): "local Ollama did not answer in time",
    (SOURCE_OLLAMA, STATUS_ERROR): "local Ollama returned an error",
    (SOURCE_HUGGINGFACE, STATUS_NOT_FOUND): "not found on Hugging Face",
    (SOURCE_HUGGINGFACE, STATUS_UNREACHABLE): "Hugging Face could not be reached",
    (SOURCE_HUGGINGFACE, STATUS_TIMEOUT): "Hugging Face did not answer in time",
    (SOURCE_HUGGINGFACE, STATUS_ERROR): "Hugging Face returned an error",
}


@dataclass
class Row:
    """One output row: a heading and one or more (label, explanation) entries."""

    heading: str
    entries: list[tuple[str, str | None]] = field(default_factory=list)
    style: str = ""


def _family_rows(parsed: ParsedName) -> list[tuple[str, str | None]]:
    """Fold versions into the family they follow: "Gemma 4 (Google)"."""
    groups: list[dict[str, object]] = []
    for token in parsed.tokens:
        reading = token.reading
        if reading is None:
            continue
        if reading.category == "family":
            match = next((g for g in groups if g["value"] == reading.value), None)
            if match is None:
                groups.append({"value": reading.value, "reading": reading, "versions": []})
            else:
                groups.append(groups.pop(groups.index(match)))
        elif reading.category == "version" and groups:
            versions = groups[-1]["versions"]
            assert isinstance(versions, list)
            versions.append(reading)
    entries: list[tuple[str, str | None]] = []
    for group in groups:
        reading, versions = group["reading"], group["versions"]
        assert isinstance(reading, Reading)
        assert isinstance(versions, list)
        text = " ".join([reading.label, *(version.label for version in versions)])
        if reading.maker:
            text = f"{text} ({reading.maker})"
        notes = [reading.explanation, *(version.explanation for version in versions)]
        entries.append((text, " ".join(notes)))
    return entries


def name_rows(parsed: ParsedName, dictionary: Dictionary | None = None) -> list[Row]:
    """Rows for everything the name itself says, in dictionary category order."""
    dictionary = dictionary or get_dictionary()
    has_family = parsed.family is not None
    by_heading: dict[str, Row] = {}

    def row(heading: str) -> Row:
        return by_heading.setdefault(heading, Row(heading))

    readings = [token.reading for token in parsed.tokens if token.reading is not None]
    readings += [item.reading for item in parsed.implied]
    for category in dictionary.categories.values():
        if category.id == "family":
            entries = _family_rows(parsed)
            if entries:
                row(category.heading).entries.extend(entries)
            continue
        if category.id == "version" and has_family:
            continue
        for reading in readings:
            if reading.category == category.id:
                entry = (reading.label, reading.explanation)
                if entry not in row(category.heading).entries:
                    row(category.heading).entries.append(entry)

    rows = [found for found in by_heading.values() if found.entries]
    for token in parsed.ambiguous:
        options = [
            (f"{reading.label} [{dictionary.categories[reading.category].heading.lower()}]", None)
            for reading in token.readings
        ]
        first, *rest = options
        entries = [
            (f"{token.text}: {first[0]}", None),
            *((f"or {text}", note) for text, note in rest),
        ]
        rows.append(Row("Ambiguous", entries, style="yellow"))
    if parsed.unknown:
        note = "Not in the token dictionary, so no meaning is guessed."
        rows.append(Row("Unknown", [(", ".join(parsed.unknown), note)], style="yellow"))
    return rows


def _name_already_says(parsed: ParsedName, fact: Fact) -> bool:
    """True when a looked-up fact only repeats what the name states outright."""
    if fact.key == "quantization":
        return str(fact.value).lower() in {value.lower() for value in parsed.values("quantization")}
    if fact.key == "parameter_count":
        shown = format_count(fact.value) if isinstance(fact.value, (int, float)) else fact.label
        return shown in parsed.values("size")
    if fact.key == "architecture":
        return fact.value in parsed.values("architecture")
    if fact.key == "hosting":
        return fact.value in parsed.values("hosting")
    if fact.key == "context_length":
        return fact.label.split()[0] in parsed.values("context")
    return False


def lookup_rows(info: ModelInfo) -> list[Row]:
    """Rows for looked-up facts, leaving out any that merely repeat the name."""
    rows: list[Row] = []
    order = list(FACT_HEADINGS)
    facts = sorted(
        info.facts, key=lambda fact: order.index(fact.key) if fact.key in order else len(order)
    )
    for fact in facts:
        if _name_already_says(info.parsed, fact):
            continue
        heading = FACT_HEADINGS.get(fact.key, fact.key.replace("_", " ").capitalize())
        rows.append(Row(heading, [(fact.label, fact.explanation)]))
    return rows


def lookup_note(info: ModelInfo) -> str | None:
    """One line saying why there are no looked-up facts, or ``None`` to stay silent."""
    reasons = [
        _MISS_REASONS[source, status]
        for source, status in info.lookups.items()
        if (source, status) in _MISS_REASONS
    ]
    return "; ".join(reasons) if reasons else None


def _kinds_line(parsed: ParsedName) -> str | None:
    if len(parsed.kinds) < 2:
        return None
    phrases = [_KIND_PHRASES.get(kind, kind) for kind in parsed.kinds]
    if parsed.tag_is_default and KIND_OLLAMA in parsed.kinds:
        index = parsed.kinds.index(KIND_OLLAMA)
        phrases[index] = f"{phrases[index]} (tag {parsed.tag})"
    return f"Could be {' or '.join(phrases)}; the name alone does not say which."


def _table(rows: list[Row], verbose: bool) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for found in rows:
        cell = Text()
        for index, (label, explanation) in enumerate(found.entries):
            if index:
                cell.append("\n" if verbose or found.heading == "Ambiguous" else ", ")
            cell.append(label, style=found.style)
            if verbose and explanation:
                cell.append(f"\n{explanation}", style="dim")
        table.add_row(Text(found.heading, style="bold" if not found.style else found.style), cell)
    return table


def render_info(info: ModelInfo, *, verbose: bool = False) -> RenderableType:
    """Build the text block for one model."""
    parsed = info.parsed
    parts: list[RenderableType] = [Text(parsed.raw, style="bold cyan")]
    kinds = _kinds_line(parsed)
    if kinds:
        parts.append(Text(kinds, style="dim"))
    parts.append(Text())
    rows = name_rows(parsed)
    if rows:
        parts.append(_table(rows, verbose))

    looked_up = lookup_rows(info)
    note = lookup_note(info)
    if looked_up:
        sources = sorted({fact.source for fact in info.facts})
        origin = ", ".join(_SOURCE_NAMES.get(source, source) for source in sources)
        heading = Text(LOOKUP_HEADING, style="bold")
        heading.append(f"  (from {origin})", style="dim")
        parts += [Text(), heading, _table(looked_up, verbose)]
    elif note and not any(status == STATUS_OK for status in info.lookups.values()):
        parts += [Text(), Text(f"Nothing looked up: {note}.", style="dim")]
    return Padding(Group(*parts), (1, 2, 0, 2))


def render_tokens(dictionary: Dictionary | None = None) -> RenderableType:
    """Build the listing for ``whatmodel tokens``."""
    dictionary = dictionary or get_dictionary()
    parts: list[RenderableType] = []
    for category in dictionary.categories.values():
        entries = dictionary.entries_in(category.id)
        if not entries:
            continue
        title = Text(category.heading, style="bold cyan")
        title.append(f"  {category.id}", style="dim")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", overflow="fold", max_width=26)
        table.add_column(overflow="fold")
        for entry in entries:
            shown = entry.examples if entry.is_regex else entry.patterns
            cell = Text(entry.label)
            cell.append(f"\n{entry.explanation}", style="dim")
            table.add_row(", ".join(shown), cell)
        parts += [Text(), title, Text(category.description, style="dim"), table]
    return Padding(Group(*parts), (0, 2, 1, 2))


def print_info(console: Console, info: ModelInfo, *, verbose: bool = False) -> None:
    console.print(render_info(info, verbose=verbose))


__all__ = [
    "FACT_HEADINGS",
    "LOOKUP_HEADING",
    "Row",
    "lookup_note",
    "lookup_rows",
    "name_rows",
    "print_info",
    "render_info",
    "render_tokens",
]
