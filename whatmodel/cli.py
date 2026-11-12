"""Command line interface.

whatmodel <name> [<name>...]   explain one or more names
whatmodel list                 explain every model installed in local Ollama
whatmodel tokens               print the token dictionary
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from typer.core import TyperGroup

from whatmodel import __version__
from whatmodel.dictionary import get_dictionary
from whatmodel.lookup import (
    DEFAULT_OLLAMA_URL,
    ModelInfo,
    OllamaUnavailableError,
    explain,
    list_installed,
)
from whatmodel.render import compare_rows, print_info, render_compare, render_tokens

_GROUP_OPTIONS = frozenset({"--help", "-h", "--version"})


class _ExplainByDefault(TyperGroup):
    """Treat ``whatmodel <name>`` as ``whatmodel explain <name>``.

    A model that happens to be called "list" or "tokens" can still be explained
    with ``whatmodel explain list``.
    """

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = ["explain", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_ExplainByDefault,
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Explain LLM model names in plain English.",
)

JsonOption = Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")]
OfflineOption = Annotated[
    bool, typer.Option("--offline", help="Read the name only; make no network requests.")
]
OllamaUrlOption = Annotated[
    str, typer.Option("--ollama-url", help="Base URL of the Ollama API.", envvar="OLLAMA_HOST")
]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Show a one-sentence explanation under each row.")
]


def _ollama_url(value: str) -> str:
    """Accept the bare ``host:port`` form that OLLAMA_HOST usually holds."""
    return value if "://" in value else f"http://{value}"


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2))


def _version(value: bool) -> None:
    if value:
        typer.echo(f"whatmodel {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
) -> None:
    """Explain LLM model names in plain English."""


def _explain_all(
    names: list[str], *, offline: bool, ollama_url: str
) -> tuple[list[ModelInfo], list[str]]:
    infos: list[ModelInfo] = []
    errors: list[str] = []
    for name in names:
        try:
            infos.append(explain(name, offline=offline, ollama_url=_ollama_url(ollama_url)))
        except ValueError as exc:
            errors.append(f"{name!r}: {exc}")
    return infos, errors


def _report(
    infos: list[ModelInfo], errors: list[str], *, as_json: bool, as_list: bool, verbose: bool
) -> None:
    if as_json:
        data = [info.to_dict() for info in infos]
        _print_json(data if as_list or len(data) != 1 else data[0])
    else:
        console = Console()
        for info in infos:
            print_info(console, info, verbose=verbose)
        if infos:
            console.print()
    for error in errors:
        typer.echo(f"whatmodel: {error}", err=True)
    if errors:
        raise typer.Exit(code=1)


@app.command("explain")
def explain_command(
    names: Annotated[list[str], typer.Argument(help="Model names to explain.")],
    as_json: JsonOption = False,
    offline: OfflineOption = False,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    verbose: VerboseOption = False,
) -> None:
    """Explain one or more model names. This is what `whatmodel <name>` runs."""
    infos, errors = _explain_all(names, offline=offline, ollama_url=ollama_url)
    _report(infos, errors, as_json=as_json, as_list=len(names) > 1, verbose=verbose)


@app.command("compare")
def compare_command(
    names: Annotated[list[str], typer.Argument(help="Two or more model names to compare.")],
    as_json: JsonOption = False,
    offline: OfflineOption = False,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
) -> None:
    """Put two or more model names side by side and mark the fields that differ."""
    if len(names) < 2:
        typer.echo("whatmodel: compare needs at least two names.", err=True)
        raise typer.Exit(code=2)
    infos, errors = _explain_all(names, offline=offline, ollama_url=ollama_url)
    if as_json:
        _print_json(
            {
                "models": [info.to_dict() for info in infos],
                "fields": [
                    {
                        "field": heading,
                        "source": "lookup" if looked_up else "name",
                        "values": values,
                        "differs": len(set(values)) > 1,
                    }
                    for heading, looked_up, values in compare_rows(infos)
                ],
            }
        )
    elif infos:
        console = Console()
        console.print(render_compare(infos))
        console.print()
    for error in errors:
        typer.echo(f"whatmodel: {error}", err=True)
    if errors:
        raise typer.Exit(code=1)


@app.command("list")
def list_command(
    as_json: JsonOption = False,
    offline: OfflineOption = False,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    verbose: VerboseOption = False,
) -> None:
    """Explain every model installed in local Ollama.

    Ollama has to be running to list what is installed. With --offline the
    names are still listed, but nothing more is looked up about them.
    """
    try:
        names = list_installed(_ollama_url(ollama_url))
    except OllamaUnavailableError as exc:
        typer.echo(f"whatmodel: {exc}. Is Ollama running?", err=True)
        raise typer.Exit(code=1) from exc
    if not names and not as_json:
        typer.echo("No models are installed in Ollama.")
        return
    infos, errors = _explain_all(names, offline=offline, ollama_url=ollama_url)
    _report(infos, errors, as_json=as_json, as_list=True, verbose=verbose)


@app.command("tokens")
def tokens_command(as_json: JsonOption = False) -> None:
    """Print the token dictionary: every token whatmodel knows and what it means."""
    dictionary = get_dictionary()
    if not as_json:
        Console().print(render_tokens(dictionary))
        return
    _print_json(
        {
            "categories": [
                {
                    "id": category.id,
                    "heading": category.heading,
                    "description": category.description,
                }
                for category in dictionary.categories.values()
            ],
            "tokens": [
                {
                    "pattern": list(entry.patterns),
                    "regex": entry.is_regex,
                    "category": entry.category,
                    "label": entry.label,
                    "explanation": entry.explanation,
                    "examples": list(entry.examples),
                }
                for entry in dictionary.entries
            ],
        }
    )


def main() -> None:
    """Entry point for the ``whatmodel`` command."""
    app()


if __name__ == "__main__":
    main()
