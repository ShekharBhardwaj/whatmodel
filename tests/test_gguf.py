"""Reading GGUF headers, and the facts built from them."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from whatmodel import explain, parse
from whatmodel.gguf import GGUFError, file_type_name, read_metadata
from whatmodel.lookup import ModelInfo, gguf_facts
from whatmodel.render import display_name, report_url

_TYPES = {"u32": (4, "I"), "u64": (10, "Q"), "f32": (6, "f"), "bool": (7, "?")}


def _string(text: str, order: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack(order + "Q", len(data)) + data


def _value(value: Any, order: str) -> bytes:
    if isinstance(value, str):
        return struct.pack(order + "I", 8) + _string(value, order)
    if isinstance(value, bool):
        return struct.pack(order + "I?", 7, value)
    if isinstance(value, int):
        return struct.pack(order + "II", 4, value)
    if isinstance(value, list):
        head = struct.pack(order + "IIQ", 9, 8, len(value))
        return head + b"".join(_string(item, order) for item in value)
    raise TypeError(type(value))


def write_gguf(path: Path, pairs: dict[str, Any], *, order: str = "<", version: int = 3) -> Path:
    body = b"".join(_string(key, order) + _value(value, order) for key, value in pairs.items())
    header = b"GGUF" + struct.pack(order + "IQQ", version, 0, len(pairs))
    path.write_bytes(header + body + b"\0" * 64)  # stand-in for the tensor data
    return path


METADATA: dict[str, Any] = {
    "general.architecture": "llama",
    "general.name": "Llama 3.2 3B Instruct",
    "general.size_label": "3B",
    "general.file_type": 15,
    "general.license": "llama3.2",
    "general.base_model.count": 1,
    "general.base_model.0.name": "Llama 3.2 3B",
    "general.base_model.0.organization": "Meta Llama",
    "llama.context_length": 131072,
    "tokenizer.ggml.tokens": [f"tok{n}" for n in range(500)],
    "tokenizer.chat_template": "{% if tools %}...{% endif %}",
    "general.languages": ["en", "de"],
}


def test_reads_the_header(tmp_path: Path) -> None:
    metadata = read_metadata(write_gguf(tmp_path / "m.gguf", METADATA))
    assert metadata["general.architecture"] == "llama"
    assert metadata["llama.context_length"] == 131072
    assert metadata["general.languages"] == ["en", "de"]
    assert file_type_name(metadata) == "Q4_K_M"
    # The vocabulary is skipped, and what comes after it is still read.
    assert "tokenizer.ggml.tokens" not in metadata
    assert "tokenizer.chat_template" in metadata


def test_reads_big_endian_and_version_2(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", METADATA, order=">", version=2)
    assert read_metadata(path)["llama.context_length"] == 131072


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not a model at all", "not a GGUF file"),
        (b"GGUF", "ends in the middle"),
        (b"GGUF" + struct.pack("<IQQ", 9, 0, 0), "version 9 is not supported"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 10**9), "implausible number"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 3) + b"ab", "ends in the"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 10**12), "implausibly long"),
    ],
)
def test_bad_files_are_rejected(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(content)
    with pytest.raises(GGUFError, match=message):
        read_metadata(path)


def test_unknown_file_type_is_not_named() -> None:
    assert file_type_name({"general.file_type": 9999}) is None
    assert file_type_name({}) is None


def test_gguf_facts() -> None:
    facts = {fact.key: fact for fact in gguf_facts(METADATA, 2_019_377_376)}
    assert {fact.source for fact in facts.values()} == {"file"}
    assert facts["quantization"].value == "Q4_K_M"
    assert facts["context_length"].label == "128k tokens"
    assert facts["parameter_count"].value == "3B"
    assert facts["architecture"].value == "dense"
    assert facts["tools"].label == "yes (judged from the chat template)"
    assert facts["license"].value == "llama3.2"
    assert facts["base_model"].label == "Llama 3.2 3B (Meta Llama)"
    assert facts["file_size"].label == "2.0 GB"


def test_expert_count_means_mixture_of_experts() -> None:
    metadata = {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 128}
    (fact,) = gguf_facts(metadata)
    assert (fact.key, fact.value) == ("architecture", "moe")


def test_nothing_is_claimed_from_an_empty_header() -> None:
    assert gguf_facts({}) == []


def test_a_file_on_disk_is_read_even_offline(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "Llama-3.2-3B-Instruct-Q4_K_M.gguf", METADATA)
    info = explain(str(path), offline=True)
    assert info.lookups == {"ollama": "skipped", "huggingface": "skipped", "file": "ok"}
    assert info.fact("context_length") is not None
    assert {fact.source for fact in info.facts} == {"file"}
    assert info.parsed.family == "llama"  # the name is still read as before


def test_a_file_name_with_no_file_behind_it_is_only_a_name() -> None:
    info = explain("Llama-3-8B-Instruct.Q4_K_M.gguf")
    assert info.facts == ()
    assert "file" not in info.lookups


def test_an_unreadable_file_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "broken-7b.gguf"
    path.write_text("not gguf", encoding="utf-8")
    info = explain(str(path))
    assert info.facts == ()
    assert info.lookups["file"] == "error"


def test_the_folder_of_a_local_file_is_never_shown(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "mystery-zzz.gguf", {})
    parsed = parse(str(path))
    assert display_name(parsed) == "mystery-zzz.gguf"
    assert tmp_path.name not in report_url(parsed)
    assert display_name(ModelInfo(parse("qwen3.5:4b")).parsed) == "qwen3.5:4b"
