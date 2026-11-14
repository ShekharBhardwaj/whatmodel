"""The library as someone using ``import whatmodel`` meets it."""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest
from rich.console import Console

import whatmodel
from whatmodel import Fact, ModelInfo, parse

LOOKED_UP = ModelInfo(
    parse("gemma4:e2b-it-qat"),
    (
        Fact("quantization", "Q4_0", "4-bit (Q4_0)", "ollama"),
        Fact("context_length", 131072, "128k tokens", "ollama"),
        Fact("parameter_count", 4628569635, "4.6B", "ollama"),
        Fact("tools", True, "yes", "ollama"),
        Fact("vision", False, "no", "ollama"),
        Fact("license", "apache-2.0", "apache-2.0", "ollama"),
        Fact("hosting", "local", "locally, installed in Ollama here", "ollama"),
        Fact("base_model", {"models": ["google/gemma-4-e2b-it"], "relation": None}, "x", "ollama"),
    ),
    {"ollama": "ok", "huggingface": "not_applicable"},
)


def test_everything_listed_is_importable() -> None:
    for name in whatmodel.__all__:
        assert hasattr(whatmodel, name), name
    assert {"parse", "explain", "explain_many", "compare", "meanings"} <= set(whatmodel.__all__)


def test_import_stays_light() -> None:
    code = (
        "import sys, whatmodel; whatmodel.parse('qwen3.5:4b'); "
        "print([m for m in ('rich', 'httpx', 'typer') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_parsed_name_prints_as_one_line() -> None:
    assert str(parse("qwen3.5:4b")) == "Qwen 3.5 (Alibaba); 4B parameters"
    assert str(parse("mystery:zzz")) == "unknown: mystery, zzz"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("llama3.1:8b", 8_000_000_000),
        ("gemma3:270m", 270_000_000),
        ("qwen2.5:0.5b", 500_000_000),
        ("Qwen3-30B-A3B", 30_000_000_000),
        ("llama3", None),
        ("gemma4:e2b", None),  # an effective size is not the parameter count
    ],
)
def test_parameters_as_a_number(name: str, expected: int | None) -> None:
    assert parse(name).parameters == expected


def test_info_exposes_what_the_name_says() -> None:
    info = LOOKED_UP
    assert (info.name, info.family, info.version) == ("gemma4:e2b-it-qat", "gemma", "4")
    assert (info.tuning, info.unknown) == ("it", ())


def test_info_prefers_looked_up_answers() -> None:
    info = LOOKED_UP
    assert info.quantization == "Q4_0"  # the name only says "qat"
    assert info.parameter_count == 4628569635
    assert info.context_length == 131072
    assert (info.supports_tools, info.supports_vision) == (True, False)
    assert info.license == "apache-2.0"
    assert info.runs == "local"
    assert info.based_on == ("google/gemma-4-e2b-it",)
    assert info.get("context_length") == 131072
    assert info.get("nonsense", "fallback") == "fallback"


def test_info_without_lookups_says_none_not_no() -> None:
    info = whatmodel.explain("llama3.1:8b-instruct-q4_K_M", offline=True)
    assert info.context_length is None
    assert info.supports_tools is None
    assert info.supports_thinking is None
    assert info.license is None
    assert info.runs is None
    assert info.based_on == ()
    # These fall back to what the name states.
    assert info.quantization == "Q4_K_M"
    assert info.parameter_count == 8_000_000_000


def test_info_prints_as_text_and_as_a_table() -> None:
    text = str(LOOKED_UP)
    assert text.splitlines()[0] == "gemma4:e2b-it-qat"
    assert "- Family: Gemma 4 (Google) [name]" in text
    assert "- Context: 128k tokens [local Ollama]" in text
    assert LOOKED_UP.summary.startswith("Gemma 4 (Google); effective 2B parameters")
    assert "128k tokens context" in LOOKED_UP.summary

    console = Console(record=True, width=100, color_system=None)
    console.print(LOOKED_UP)
    drawn = console.export_text()
    assert "│ Family" in drawn
    assert "local Ollama" in drawn


def test_compare() -> None:
    result = whatmodel.compare(
        "llama3.1:8b-instruct-q4_K_M", "llama3.1:8b-instruct-q5_K_M", offline=True
    )
    assert [model.name for model in result.models] == [
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q5_K_M",
    ]
    assert [found.field for found in result.fields] == ["Family", "Size", "Tuning", "Compression"]
    (difference,) = result.differences
    assert difference.field == "Compression"
    assert difference.looked_up is False
    assert difference.values[0] != difference.values[1]
    assert "- Compression (from the name, differs):" in str(result)
    assert result.to_dict()["fields"][0] == {
        "field": "Family",
        "source": "name",
        "values": ["Llama 3.1 (Meta)", "Llama 3.1 (Meta)"],
        "differs": False,
    }
    console = Console(record=True, width=120, color_system=None)
    console.print(result)
    assert "differs" in console.export_text()


def test_compare_needs_two_names() -> None:
    with pytest.raises(ValueError, match="at least two names"):
        whatmodel.compare("qwen3.5:4b", offline=True)


def test_meanings() -> None:
    (reading,) = whatmodel.meanings("QAT")
    assert reading.label == "quantization-aware trained"
    assert whatmodel.meanings("Q4_K_M")[0].category == "quantization"
    assert whatmodel.meanings("definitely-not-a-token") == ()
    assert {reading.category for reading in whatmodel.meanings("base")} == {"tier", "tuning"}


def test_explain_many_keeps_order_and_shares_one_client() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    names = ["gemma4:12b", "org/repo", "llama3:8b"]
    infos = whatmodel.explain_many(names, client=client)
    assert [info.name for info in infos] == names
    assert "huggingface.co" in hosts
    assert not client.is_closed  # a client the caller passed in is the caller's to close


def test_explain_many_offline_makes_no_client() -> None:
    infos = whatmodel.explain_many(["gemma4:12b", "llama3:8b"], offline=True)
    assert [info.family for info in infos] == ["gemma", "llama"]
