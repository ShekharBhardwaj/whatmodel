"""CLI and text output tests. Lookups are replaced, so nothing touches the network."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

import whatmodel
from whatmodel import Fact, ModelInfo, parse
from whatmodel.cli import app
from whatmodel.lookup import OllamaUnavailableError
from whatmodel.render import lookup_note, lookup_rows, name_rows, render_info

runner = CliRunner()


def text_of(info: ModelInfo, *, verbose: bool = False) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=100, color_system=None).print(render_info(info, verbose=verbose))
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())


def cells_of(info: ModelInfo, *, verbose: bool = False) -> list[list[str]]:
    """The table as a list of rows, each a list of cell texts."""
    lines = [line for line in text_of(info, verbose=verbose).splitlines() if "\u2502" in line]
    return [[cell.strip() for cell in line.strip("\u2502").split("\u2502")] for line in lines]


def rows_of(name: str) -> dict[str, str]:
    return {
        row.heading: " | ".join(label for label, _ in row.entries) for row in name_rows(parse(name))
    }


LOOKED_UP = ModelInfo(
    parse("gemma4:e2b-it-qat"),
    (
        Fact("quantization", "Q4_0", "4-bit (Q4_0)", "ollama", "Compressed."),
        Fact("context_length", 131072, "128k tokens", "ollama"),
        Fact("tools", True, "yes", "ollama"),
        Fact("license", "Apache License, Version 2.0", "Apache License, Version 2.0", "ollama"),
        Fact("hosting", "local", "locally, installed in Ollama here", "ollama"),
    ),
    {"ollama": "ok", "huggingface": "not_applicable"},
)


# ------------------------------------------------------------------ render ----


def test_layout() -> None:
    assert text_of(LOOKED_UP) == "\n".join(
        [
            "",
            "gemma4:e2b-it-qat",
            "┌─────────────┬────────────────────────────────────┬──────────────┐",
            "│ Field       │ Value                              │ Source       │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Family      │ Gemma 4 (Google)                   │ name         │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Size        │ effective 2B parameters            │ name         │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Tuning      │ instruction tuned (ready for chat) │ name         │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Compression │ quantization-aware trained         │ name         │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Runs        │ locally, installed in Ollama here  │ local Ollama │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Compression │ 4-bit (Q4_0)                       │ local Ollama │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Context     │ 128k tokens                        │ local Ollama │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ Tools       │ yes                                │ local Ollama │",
            "├─────────────┼────────────────────────────────────┼──────────────┤",
            "│ License     │ Apache License, Version 2.0        │ local Ollama │",
            "└─────────────┴────────────────────────────────────┴──────────────┘",
        ]
    )


def test_every_row_says_where_its_fact_came_from() -> None:
    rows = cells_of(LOOKED_UP)
    assert rows[0] == ["Field", "Value", "Source"]
    sources = [row[2] for row in rows[1:]]
    # Name rows come first, looked-up rows after, and the two never interleave.
    assert sources == ["name"] * 4 + ["local Ollama"] * 5
    assert ["Family", "Gemma 4 (Google)", "name"] in rows
    assert ["Context", "128k tokens", "local Ollama"] in rows
    # Where it runs is only ever a looked-up fact here, never read off the name.
    assert ["Runs", "locally, installed in Ollama here", "local Ollama"] in rows


def test_assumed_default_tag_is_a_note_not_a_row() -> None:
    # Whether a bare name is an Ollama name at all is not something it can prove.
    info = ModelInfo(parse("qwen-max-0725"))
    assert "Build" not in rows_of("qwen-max-0725")
    assert "Runs" not in rows_of("qwen-max-0725")
    assert "Pulled with Ollama, it would mean latest." in text_of(info)
    assert "Pulled with Ollama" not in text_of(ModelInfo(parse("llama3:latest")))


def test_offline_output_has_only_name_rows() -> None:
    info = ModelInfo(parse("qwen3.5:4b"), (), {"ollama": "skipped", "huggingface": "skipped"})
    assert {row[2] for row in cells_of(info)[1:]} == {"name"}
    assert "Nothing looked up" not in text_of(info)
    assert "Qwen 3.5 (Alibaba)" in text_of(info)


def test_failed_lookup_leaves_one_quiet_line() -> None:
    info = ModelInfo(parse("qwen3.5:4b"), (), {"ollama": "unreachable", "huggingface": "timeout"})
    assert lookup_note(info) == "local Ollama is not running; Hugging Face did not answer in time"
    assert "Nothing looked up: local Ollama is not running" in text_of(info)
    assert ["Size", "4B parameters", "name"] in cells_of(info)


def test_lookup_facts_that_repeat_the_name_are_left_out() -> None:
    info = ModelInfo(
        parse("Llama-3-8B-Instruct-Q4_K_M:cloud"),
        (
            Fact("quantization", "Q4_K_M", "4-bit", "ollama"),
            Fact("parameter_count", 8030261248, "8B", "ollama"),
            Fact("hosting", "cloud", "remotely", "ollama"),
            Fact("tools", False, "no", "ollama"),
        ),
        {"ollama": "ok"},
    )
    assert [row.heading for row in lookup_rows(info)] == ["Tools"]


@pytest.mark.parametrize(
    ("name", "heading", "expected"),
    [
        ("kimi-k3.1-pro:cloud", "Family", "Kimi K3.1 (Moonshot AI)"),
        ("kimi-k3.1-pro:cloud", "Runs", "in the cloud (remote, not on your machine)"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Family", "Mistral (Mistral AI)"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Version", "v0.3"),
        (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Family",
            "DeepSeek R1 (DeepSeek) | Qwen (Alibaba)",
        ),
        ("hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M", "Family", "Qwen 3 (Alibaba)"),
        ("Qwen3-30B-A3B", "Size", "30B parameters | 3B active per token"),
        ("Qwen3-30B-A3B", "Architecture", "mixture of experts"),
        ("llama3:latest", "Build", "latest (the default build)"),
        ("mystery:v2", "Version", "v2"),
        ("some-weird-xyz:abc", "Unknown", "some-weird-xyz, abc"),
    ],
)
def test_rows(name: str, heading: str, expected: str) -> None:
    assert rows_of(name)[heading] == expected


def test_ambiguous_token_shows_every_reading() -> None:
    rows = cells_of(ModelInfo(parse("qwen-max-0725")))
    assert ["Ambiguous", "0725: dated 07-25 (month-day) [build]", "name"] in rows
    assert ["", "or released 2025-07 [build]", ""] in rows


def test_ambiguous_name_format_shows_both_readings() -> None:
    text = text_of(ModelInfo(parse("bartowski/Qwen3-30B-A3B-Instruct-GGUF")))
    assert "Read as a Hugging Face repo or an Ollama model (tag latest)." in text


def test_lookup_that_settles_the_format_drops_the_both_readings_line() -> None:
    parsed = parse("bartowski/Qwen3-30B-A3B-Instruct-GGUF")
    settled = ModelInfo(parsed, (), {"ollama": "not_found", "huggingface": "ok"})
    assert "Read as" not in text_of(settled)
    unsettled = ModelInfo(parsed, (), {"ollama": "not_found", "huggingface": "not_found"})
    assert "Read as" in text_of(unsettled)


def test_name_with_two_families() -> None:
    name = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
    rows = rows_of(name)
    assert rows["Family"] == "Qwen 3.5 (Alibaba) | Claude 4.6 (Anthropic)"
    assert rows["Version"] == "v2"  # the trailing version belongs to neither family
    assert "Runs" not in rows  # a hosted family named as a source implies nothing
    cells = cells_of(ModelInfo(parse(name)))
    assert ["Family", "Qwen 3.5 (Alibaba)", "name"] in cells
    assert ["", "Claude 4.6 (Anthropic)", ""] in cells  # each family on its own line


def test_verbose_adds_a_column_of_explanations() -> None:
    assert cells_of(LOOKED_UP, verbose=True)[0] == ["Field", "Value", "Source", "What it means"]
    plain, verbose = text_of(LOOKED_UP), text_of(LOOKED_UP, verbose=True)
    assert "Trained with compression in the" not in plain
    assert "Trained with compression in the" in verbose
    assert "Compressed." in verbose


# --------------------------------------------------------------------- cli ----


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"whatmodel {whatmodel.__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "tokens" in result.output


def test_explain_one_name_offline() -> None:
    result = runner.invoke(app, ["qwen3.5:4b", "--offline"])
    assert result.exit_code == 0
    assert "Qwen 3.5 (Alibaba)" in result.output
    assert "4B parameters" in result.output
    assert "Ollama" not in result.output


def test_flags_work_before_the_name_too() -> None:
    result = runner.invoke(app, ["--offline", "--json", "qwen3.5:4b"])
    assert result.exit_code == 0
    assert json.loads(result.output)["input"] == "qwen3.5:4b"


def test_explain_several_names() -> None:
    result = runner.invoke(app, ["gemma4:12b", "llama3", "--offline"])
    assert result.exit_code == 0
    assert "Gemma 4 (Google)" in result.output
    assert "Llama 3 (Meta)" in result.output


def test_json_for_one_name_is_an_object_with_sources() -> None:
    result = runner.invoke(app, ["gemma4:e2b-it-qat", "--offline", "--json"])
    data = json.loads(result.output)
    assert data["input"] == "gemma4:e2b-it-qat"
    assert {token["source"] for token in data["tokens"]} == {"name"}
    assert data["facts"] == []
    assert data["lookups"] == {"ollama": "skipped", "huggingface": "skipped"}


def test_json_for_several_names_is_a_list() -> None:
    result = runner.invoke(app, ["gemma4:12b", "llama3", "--offline", "--json"])
    assert [item["input"] for item in json.loads(result.output)] == ["gemma4:12b", "llama3"]


def test_lookup_results_reach_the_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_explain(name: str, **options: object) -> ModelInfo:
        calls.append({"name": name, **options})
        return LOOKED_UP

    monkeypatch.setattr("whatmodel.cli.explain", fake_explain)
    result = runner.invoke(app, ["gemma4:e2b-it-qat", "--ollama-url", "box:9999"])
    assert result.exit_code == 0
    assert "local Ollama" in result.output
    assert "128k tokens" in result.output
    assert calls == [
        {"name": "gemma4:e2b-it-qat", "offline": False, "ollama_url": "http://box:9999"}
    ]

    result = runner.invoke(app, ["gemma4:e2b-it-qat", "--json"])
    facts = json.loads(result.output)["facts"]
    assert {fact["source"] for fact in facts} == {"ollama"}


def test_explicit_explain_command_reaches_reserved_names() -> None:
    result = runner.invoke(app, ["explain", "list", "--offline"])
    assert result.exit_code == 0
    assert "Unknown" in result.output


def test_empty_name_is_reported_and_others_still_print() -> None:
    result = runner.invoke(app, ["", "llama3", "--offline"])
    assert result.exit_code == 1
    assert "Llama 3 (Meta)" in result.output
    assert "model name is empty" in result.output


def test_unknown_tokens_do_not_fail_the_command() -> None:
    result = runner.invoke(app, ["some-weird-xyz:abc", "--offline"])
    assert result.exit_code == 0
    assert "some-weird-xyz, abc" in result.output


def test_tokens_command() -> None:
    result = runner.invoke(app, ["tokens"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for expected in ("Compression", "quantization-aware trained", "4b, 12B, 1.5b", "bartowski"):
        assert expected in result.output
    for header in ("Token", "Meaning", "What it means"):
        assert header in result.output
    assert "\u250c" in result.output  # drawn as a grid table
    assert "(?P<" not in result.output  # examples are shown, never raw regexes


def test_tokens_command_json() -> None:
    data = json.loads(runner.invoke(app, ["tokens", "--json"]).output)
    assert "family" in {category["id"] for category in data["categories"]}
    assert any(token["pattern"] == ["qat"] for token in data["tokens"])


def test_list_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("whatmodel.cli.list_installed", lambda url: ["gemma4:12b", "llama3:latest"])
    result = runner.invoke(app, ["list", "--offline"])
    assert result.exit_code == 0
    assert "gemma4:12b" in result.output
    assert "Llama 3 (Meta)" in result.output

    result = runner.invoke(app, ["list", "--offline", "--json"])
    assert [item["input"] for item in json.loads(result.output)] == ["gemma4:12b", "llama3:latest"]


def test_list_command_with_nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("whatmodel.cli.list_installed", lambda url: [])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models are installed" in result.output
    assert json.loads(runner.invoke(app, ["list", "--json"]).output) == []


def test_list_command_when_ollama_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(url: str) -> list[str]:
        raise OllamaUnavailableError(f"could not list models from {url} (unreachable)")

    monkeypatch.setattr("whatmodel.cli.list_installed", down)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "Is Ollama running?" in result.output
