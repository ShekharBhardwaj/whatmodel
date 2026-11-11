"""Text output. Layout lives here; every word about a token comes from tokens.yaml.

Facts from the name and facts from lookups are kept in two separate blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich import box
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
    #: Put each entry on its own line instead of joining them with commas.
    stacked: bool = False
    #: Where the row's facts come from: "name", or a lookup source.
    source: str = "name"


def _family_rows(parsed: ParsedName) -> tuple[list[tuple[str, str | None]], set[int]]:
    """Fold a version into the family it directly follows: "Gemma 4 (Google)".

    Returns the entries and the positions of the version tokens that were used.
    A version elsewhere in the name, such as a trailing "v2", belongs to the
    model as a whole and is left for the Version row.
    """
    groups: list[dict[str, object]] = []
    attached: set[int] = set()
    open_group: dict[str, object] | None = None
    for position, token in enumerate(parsed.tokens):
        reading = token.reading
        if reading is not None and reading.category == "family":
            open_group = next((g for g in groups if g["value"] == reading.value), None)
            if open_group is None:
                open_group = {"value": reading.value, "reading": reading, "versions": []}
                groups.append(open_group)
        elif reading is not None and reading.category == "version" and open_group is not None:
            versions = open_group["versions"]
            assert isinstance(versions, list)
            versions.append(reading)
            attached.add(position)
        else:
            open_group = None
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
    return entries, attached


def name_rows(parsed: ParsedName, dictionary: Dictionary | None = None) -> list[Row]:
    """Rows for everything the name itself says, in dictionary category order."""
    dictionary = dictionary or get_dictionary()
    by_heading: dict[str, Row] = {}

    def row(heading: str) -> Row:
        return by_heading.setdefault(heading, Row(heading))

    family_entries, attached = _family_rows(parsed)
    readings = [
        token.reading
        for position, token in enumerate(parsed.tokens)
        if token.reading is not None and position not in attached
    ]
    readings += [item.reading for item in parsed.implied]
    for category in dictionary.categories.values():
        if category.id == "family":
            if family_entries:
                found = row(category.heading)
                found.entries.extend(family_entries)
                found.stacked = True
            continue
        for reading in readings:
            if reading.category == category.id:
                entry = (reading.label, reading.explanation)
                if entry not in row(category.heading).entries:
                    row(category.heading).entries.append(entry)

    rows = [found for found in by_heading.values() if found.entries]
    for token in parsed.ambiguous:
        options = [
            f"{reading.label} [{dictionary.categories[reading.category].heading.lower()}]"
            for reading in token.readings
        ]
        entries = [
            (f"{token.text}: {options[0]}", None),
            *((f"or {text}", None) for text in options[1:]),
        ]
        rows.append(Row("Ambiguous", entries, style="yellow", stacked=True))
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
        rows.append(Row(heading, [(fact.label, fact.explanation)], source=fact.source))
    return rows


def lookup_note(info: ModelInfo) -> str | None:
    """One line saying why there are no looked-up facts, or ``None`` to stay silent."""
    reasons = [
        _MISS_REASONS[source, status]
        for source, status in info.lookups.items()
        if (source, status) in _MISS_REASONS
    ]
    return "; ".join(reasons) if reasons else None


def _kinds_line(info: ModelInfo) -> str | None:
    """Name the possible formats, unless a lookup has settled which one it is."""
    parsed = info.parsed
    if len(parsed.kinds) < 2:
        return None
    found = [source for source, status in info.lookups.items() if status == STATUS_OK]
    if len(found) == 1:
        return None
    phrases = [_KIND_PHRASES.get(kind, kind) for kind in parsed.kinds]
    if parsed.tag_is_default and KIND_OLLAMA in parsed.kinds:
        index = parsed.kinds.index(KIND_OLLAMA)
        phrases[index] = f"{phrases[index]} (tag {parsed.tag})"
    return " or ".join(phrases)


def _add_rows(table: Table, rows: list[Row], verbose: bool) -> None:
    for found in rows:
        joiner = "\n" if found.stacked else ", "
        cells = [
            Text(found.heading, style=found.style or "cyan"),
            Text(joiner.join(label for label, _ in found.entries), style=found.style),
            Text(_SOURCE_NAMES.get(found.source, found.source), style="dim"),
        ]
        if verbose:
            notes = [explanation for _, explanation in found.entries if explanation]
            cells.append(Text("\n".join(notes), style="dim"))
        table.add_row(*cells)


def render_info(info: ModelInfo, *, verbose: bool = False) -> RenderableType:
    """Build one model's table: a row per fact, with a column saying where it came from.

    Rows read from the name come first. Looked-up rows follow after a divider,
    and the Source column keeps the two apart on every row.
    """
    parsed = info.parsed
    table = Table(
        title=parsed.raw,
        title_style="bold",
        title_justify="left",
        box=box.SQUARE,
        show_lines=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_column("Source", no_wrap=True)
    if verbose:
        table.add_column("What it means", overflow="fold")

    _add_rows(table, name_rows(parsed), verbose)
    looked_up = lookup_rows(info)
    if looked_up:
        table.add_section()
        _add_rows(table, looked_up, verbose)

    captions: list[str] = []
    kinds = _kinds_line(info)
    if kinds:
        captions.append(f"Read as {kinds}.")
    note = lookup_note(info)
    if note and not looked_up and STATUS_OK not in info.lookups.values():
        captions.append(f"Nothing looked up: {note}.")
    notes = [Text(caption, style="dim") for caption in captions]
    return Padding(Group(table, *notes), (1, 0, 0, 0))


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
    "Row",
    "lookup_note",
    "lookup_rows",
    "name_rows",
    "print_info",
    "render_info",
    "render_tokens",
]
