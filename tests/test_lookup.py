"""Layer 2 tests. All HTTP is mocked; conftest.py blocks real sockets."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from whatmodel import explain
from whatmodel.lookup import (
    OllamaUnavailableError,
    format_bytes,
    format_context,
    format_count,
    huggingface_facts,
    list_installed,
    ollama_facts,
)

Handler = Callable[[httpx.Request], httpx.Response]

# Shaped like real responses from Ollama 0.40 and the Hub, trimmed.
OLLAMA_SHOW: dict[str, Any] = {
    "license": "\n                    Apache License\n      Version 2.0, January 2004\n",
    "parameters": "temperature 1",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gemma4",
        "families": ["gemma4"],
        "parameter_size": "4.6B",
        "quantization_level": "Q4_0",
    },
    "model_info": {
        "general.architecture": "gemma4",
        "general.parameter_count": 4628569635,
        "gemma4.context_length": 131072,
    },
    "capabilities": ["completion", "vision", "tools", "thinking"],
}
OLLAMA_TAGS: dict[str, Any] = {
    "models": [
        {"name": "gemma4:e2b-it-qat", "model": "gemma4:e2b-it-qat", "size": 4336358185},
        {"name": "llama3:latest", "model": "llama3:latest", "size": 4661224676},
    ]
}
HUB_GGUF: dict[str, Any] = {
    "id": "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
    "pipeline_tag": "text-generation",
    "tags": ["gguf", "license:apache-2.0"],
    "cardData": {"license": "apache-2.0"},
    "gguf": {
        "total": 30532122624,
        "architecture": "qwen3moe",
        "context_length": 32768,
        "chat_template": "{%- if tools %}...{%- endif %}",
    },
    "siblings": [
        {"rfilename": "README.md", "size": 100},
        {"rfilename": "Qwen_Qwen3-30B-A3B-IQ2_M.gguf", "size": 10430210720},
        {"rfilename": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf", "size": 18556686912},
    ],
}
HUB_SAFETENSORS: dict[str, Any] = {
    "id": "Qwen/Qwen3-4B",
    "pipeline_tag": "text-generation",
    "tags": ["safetensors", "license:apache-2.0"],
    "cardData": {},
    "safetensors": {"parameters": {"BF16": 4022468096}, "total": 4022468096},
    "config": {"tokenizer_config": {"chat_template": "plain template"}},
    "gguf": None,
}


class Recorder:
    """A mock transport that records every request it serves."""

    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


def routes(show: Any = None, tags: Any = None, hub: Any = None, *, missing: int = 404) -> Recorder:
    """Serve the given bodies; anything left as ``None`` answers "not found"."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = None
        if path == "/api/show":
            body = show
        elif path == "/api/tags":
            body = tags
        elif path.startswith("/api/models/"):
            body = hub
        if body is None:
            return httpx.Response(missing, json={"error": "not found"})
        return httpx.Response(200, json=body)

    return Recorder(handler)


def failing(error: Exception) -> Recorder:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return Recorder(handler)


def facts_of(info: Any) -> dict[str, tuple[Any, str]]:
    return {fact.key: (fact.value, fact.source) for fact in info.facts}


# ------------------------------------------------------------------ ollama ----


def test_ollama_facts_come_from_show_and_tags() -> None:
    recorder = routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS)
    info = explain("gemma4:e2b-it-qat", client=recorder.client())

    assert info.lookups == {"ollama": "ok", "huggingface": "not_applicable"}
    assert facts_of(info) == {
        "quantization": ("Q4_0", "ollama"),
        "context_length": (131072, "ollama"),
        "parameter_count": (4628569635, "ollama"),
        "architecture": ("dense", "ollama"),
        "tools": (True, "ollama"),
        "vision": (True, "ollama"),
        "thinking": (True, "ollama"),
        "license": ("Apache License, Version 2.0", "ollama"),
        "file_size": (4336358185, "ollama"),
    }
    labels = {fact.key: fact.label for fact in info.facts}
    assert labels["context_length"] == "128k tokens"
    assert labels["parameter_count"] == "4.6B"
    assert labels["file_size"] == "4.3 GB"
    assert labels["quantization"] == "4-bit (Q4_0)"  # explained by tokens.yaml

    show_request = recorder.requests[0]
    assert (show_request.method, show_request.url.path) == ("POST", "/api/show")
    assert json.loads(show_request.content) == {"model": "gemma4:e2b-it-qat"}
    assert recorder.urls == [
        "http://localhost:11434/api/show",
        "http://localhost:11434/api/tags",
    ]


def test_ollama_url_is_configurable() -> None:
    recorder = routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS)
    explain("llama3", ollama_url="http://box:9999/", client=recorder.client())
    assert recorder.urls == ["http://box:9999/api/show", "http://box:9999/api/tags"]
    assert json.loads(recorder.requests[0].content) == {"model": "llama3:latest"}


def test_default_tag_matches_the_listed_model() -> None:
    info = explain("llama3", client=routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS).client())
    assert facts_of(info)["file_size"] == (4661224676, "ollama")


def test_model_not_installed_is_skipped_quietly() -> None:
    recorder = routes()
    info = explain("qwen3.5:4b", client=recorder.client())
    assert info.lookups == {"ollama": "not_found", "huggingface": "not_applicable"}
    assert info.facts == ()
    assert info.parsed.size == "4B"  # Layer 1 is still all there
    assert recorder.urls == ["http://localhost:11434/api/show"]


def test_mixture_of_experts_is_read_from_gguf_metadata() -> None:
    show = {
        "details": {"format": "gguf"},
        "model_info": {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 128},
    }
    (fact,) = [fact for fact in ollama_facts(show) if fact.key == "architecture"]
    assert (fact.value, fact.label) == ("moe", "moe: mixture of experts")


def test_dense_is_not_claimed_without_full_metadata() -> None:
    show = {"details": {"format": "safetensors"}, "model_info": {"general.architecture": "x"}}
    assert [fact for fact in ollama_facts(show) if fact.key == "architecture"] == []


def test_capabilities_are_not_guessed_when_absent() -> None:
    keys = {fact.key for fact in ollama_facts({"details": {"quantization_level": "Q8_0"}})}
    assert keys == {"quantization"}


def test_remote_model_is_reported_as_hosted() -> None:
    facts = ollama_facts({}, {"remote_host": "https://ollama.com", "size": 0})
    assert [(fact.key, fact.value) for fact in facts] == [("hosting", "cloud")]


def test_unknown_quantization_level_is_passed_through_unexplained() -> None:
    (fact,) = ollama_facts({"details": {"quantization_level": "nvfp4"}})
    assert (fact.value, fact.label, fact.explanation) == ("nvfp4", "nvfp4", None)


# ------------------------------------------------------------ hugging face ----


def test_hugging_face_is_used_for_org_repo_names() -> None:
    recorder = routes(hub=HUB_GGUF)
    info = explain("bartowski/Qwen_Qwen3-30B-A3B-GGUF", client=recorder.client())

    assert info.lookups == {"ollama": "not_found", "huggingface": "ok"}
    assert facts_of(info) == {
        "context_length": (32768, "huggingface"),
        "parameter_count": (30532122624, "huggingface"),
        "tools": (True, "huggingface"),
        "license": ("apache-2.0", "huggingface"),
        "file_size": ({"min": 10430210720, "max": 18556686912, "files": 2}, "huggingface"),
    }
    assert recorder.urls[-1] == (
        "https://huggingface.co/api/models/bartowski/Qwen_Qwen3-30B-A3B-GGUF?blobs=true"
    )


def test_quantization_in_the_name_picks_the_matching_file() -> None:
    recorder = routes(hub=HUB_GGUF)
    info = explain("hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M", client=recorder.client())
    fact = info.fact("file_size")
    assert fact is not None
    assert fact.value == 18556686912
    assert fact.label == "18.6 GB (Qwen_Qwen3-30B-A3B-Q4_K_M.gguf)"
    assert json.loads(recorder.requests[0].content) == {
        "model": "hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M"
    }


def test_safetensors_repo() -> None:
    facts = {fact.key: fact for fact in huggingface_facts(HUB_SAFETENSORS)}
    assert facts["quantization"].value == "BF16"
    assert facts["quantization"].label == "BF16: 16-bit floats (not compressed)"
    assert facts["parameter_count"].label == "4B"
    assert facts["tools"].value is False
    assert facts["license"].value == "apache-2.0"
    assert "file_size" not in facts
    assert "context_length" not in facts


def test_url_goes_to_hugging_face_only() -> None:
    recorder = routes(hub=HUB_SAFETENSORS)
    info = explain("https://huggingface.co/Qwen/Qwen3-4B", client=recorder.client())
    assert info.lookups == {"ollama": "not_applicable", "huggingface": "ok"}
    assert recorder.urls == ["https://huggingface.co/api/models/Qwen/Qwen3-4B?blobs=true"]


def test_hub_answers_401_for_a_missing_repo() -> None:
    info = explain("nobody/nothing", client=routes(missing=401).client())
    assert info.lookups == {"ollama": "not_found", "huggingface": "not_found"}
    assert info.facts == ()


# ------------------------------------------------------------------- merge ----


def test_sources_merge_and_ollama_wins() -> None:
    recorder = routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS, hub=HUB_GGUF)
    info = explain("bartowski/Qwen_Qwen3-30B-A3B-GGUF", client=recorder.client())
    assert info.lookups == {"ollama": "ok", "huggingface": "ok"}
    merged = facts_of(info)
    assert merged["context_length"] == (131072, "ollama")
    assert merged["license"] == ("Apache License, Version 2.0", "ollama")
    # Only the Hub knew the file sizes, so that fact comes from there.
    assert merged["file_size"][1] == "huggingface"
    assert len(merged) == len(info.facts)


def test_to_dict_stamps_every_fact_with_its_source() -> None:
    info = explain("gemma4:e2b-it-qat", client=routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS).client())
    data = info.to_dict()
    assert data["lookups"] == {"ollama": "ok", "huggingface": "not_applicable"}
    assert {fact["source"] for fact in data["facts"]} == {"ollama"}
    assert {token["source"] for token in data["tokens"]} == {"name"}
    assert data["facts"][0] == {
        "key": "quantization",
        "value": "Q4_0",
        "label": "4-bit (Q4_0)",
        "source": "ollama",
        "explanation": "Weights compressed to roughly 4 bits each with llama.cpp's original, "
        "simpler method.",
    }
    json.dumps(data)


# ------------------------------------------------- offline and failure paths ----


def test_offline_makes_no_request_at_all() -> None:
    recorder = routes(show=OLLAMA_SHOW, tags=OLLAMA_TAGS, hub=HUB_GGUF)
    info = explain("bartowski/Qwen_Qwen3-30B-A3B-GGUF", offline=True, client=recorder.client())
    assert recorder.requests == []
    assert info.facts == ()
    assert info.lookups == {"ollama": "skipped", "huggingface": "skipped"}
    assert info.parsed.active_params == "3B"


def test_offline_does_not_even_build_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> httpx.Client:
        raise AssertionError("offline mode must not create an HTTP client")

    monkeypatch.setattr("whatmodel.lookup._new_client", explode)
    assert explain("qwen3.5:4b", offline=True).lookups["ollama"] == "skipped"


def test_file_names_are_never_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("whatmodel.lookup._new_client", lambda: pytest.fail("no client expected"))
    info = explain("Llama-3-8B-Instruct.Q4_K_M.gguf")
    assert info.lookups == {"ollama": "not_applicable", "huggingface": "not_applicable"}
    assert info.parsed.quantization == "Q4_K_M"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (httpx.ConnectTimeout("too slow"), "timeout"),
        (httpx.ReadTimeout("too slow"), "timeout"),
        (httpx.ConnectError("refused"), "unreachable"),
        (httpx.RemoteProtocolError("garbage"), "unreachable"),
        (httpx.DecodingError("bad encoding"), "error"),
    ],
)
def test_failures_are_skipped_and_layer_one_survives(error: Exception, status: str) -> None:
    info = explain("bartowski/Qwen3-30B-A3B-Instruct-GGUF", client=failing(error).client())
    assert info.lookups == {"ollama": status, "huggingface": status}
    assert info.facts == ()
    assert (info.parsed.org, info.parsed.size) == ("bartowski", "30B")


def test_timeout_values_are_short_and_passed_to_httpx() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(404)

    explain("a/b", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [timeout["connect"] for timeout in seen] == [2.0, 5.0]
    seen.clear()
    explain("a/b", timeout=0.5, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [timeout["read"] for timeout in seen] == [0.5, 0.5]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a", "list"]),
    ],
)
def test_bad_responses_are_skipped(response: httpx.Response) -> None:
    recorder = Recorder(lambda request: response)
    info = explain("qwen3.5:4b", client=recorder.client())
    assert info.lookups["ollama"] == "error"
    assert info.facts == ()


def test_junk_fields_do_not_crash() -> None:
    show = {"details": "nope", "model_info": [1], "capabilities": "tools", "license": 7}
    hub = {"gguf": "x", "safetensors": [], "cardData": 3, "tags": "t", "siblings": [1, {"size": 2}]}
    assert ollama_facts(show, {"size": "big"}) == []
    assert huggingface_facts(hub) == []
    info = explain("a/b", client=routes(show=show, tags={"models": "x"}, hub=hub).client())
    assert info.lookups == {"ollama": "ok", "huggingface": "ok"}
    assert info.facts == ()


def test_bad_ollama_url_is_skipped() -> None:
    info = explain("qwen3.5:4b", ollama_url="not a url", client=routes().client())
    assert info.lookups["ollama"] in {"error", "unreachable", "not_found"}
    assert info.facts == ()


# -------------------------------------------------------------------- list ----


def test_list_installed() -> None:
    recorder = routes(tags=OLLAMA_TAGS)
    assert list_installed(client=recorder.client()) == ["gemma4:e2b-it-qat", "llama3:latest"]
    assert recorder.urls == ["http://localhost:11434/api/tags"]


def test_list_installed_raises_when_ollama_is_down() -> None:
    with pytest.raises(OllamaUnavailableError, match="unreachable"):
        list_installed(client=failing(httpx.ConnectError("refused")).client())


# -------------------------------------------------------------- formatting ----


@pytest.mark.parametrize(
    ("count", "text"),
    [(4628569635, "4.6B"), (268098176, "268M"), (8030261248, "8B"), (1e12, "1T"), (950, "950")],
)
def test_format_count(count: float, text: str) -> None:
    assert format_count(count) == text


@pytest.mark.parametrize(
    ("tokens", "text"),
    [(131072, "128k tokens"), (8192, "8k tokens"), (200000, "200k tokens"), (512, "512 tokens")],
)
def test_format_context(tokens: int, text: str) -> None:
    assert format_context(tokens) == text


@pytest.mark.parametrize(
    ("size", "text"), [(4336358185, "4.3 GB"), (291554930, "291.6 MB"), (12, "12 B")]
)
def test_format_bytes(size: int, text: str) -> None:
    assert format_bytes(size) == text
