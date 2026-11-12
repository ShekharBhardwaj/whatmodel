"""Layer 1 tests: reading names offline. No network (see conftest.py)."""

from __future__ import annotations

import json

import pytest

from whatmodel import ParsedName, parse

# The cases the project spec requires, with the fields each one must produce.
SPEC_CASES: list[tuple[str, dict[str, object]]] = [
    (
        "kimi-k3.1-pro:cloud",
        {"family": "kimi", "version": "k3.1", "tier": "pro", "hosting": "cloud"},
    ),
    (
        "deepseek-v4.1-flash:cloud",
        {"family": "deepseek", "version": "v4.1", "tier": "flash", "hosting": "cloud"},
    ),
    ("qwen3.5:4b", {"family": "qwen", "version": "3.5", "size": "4B"}),
    ("gemma4:12b", {"family": "gemma", "version": "4", "size": "12B"}),
    ("qwen3.8:27b", {"family": "qwen", "version": "3.8", "size": "27B"}),
    ("gemma4:e4b", {"effective_size": "4B"}),
    ("gemma4:e2b-it-qat", {"effective_size": "2B", "tuning": "it", "quantization": "qat"}),
    (
        "bartowski/Qwen3-30B-A3B-Instruct-GGUF",
        {"org": "bartowski", "size": "30B", "active_params": "3B", "architecture": "moe"},
    ),
    (
        "Llama-3-8B-Instruct.Q4_K_M.gguf",
        {"size": "8B", "tuning": "instruct", "quantization": "Q4_K_M", "format": "gguf"},
    ),
    (
        "llama3",
        {"family": "llama", "version": "3", "tag": "latest", "tag_is_default": True},
    ),
]


@pytest.mark.parametrize(("name", "expected"), SPEC_CASES, ids=[case[0] for case in SPEC_CASES])
def test_spec_cases(name: str, expected: dict[str, object]) -> None:
    parsed = parse(name)
    actual = {field: getattr(parsed, field) for field in expected}
    assert actual == expected
    assert parsed.unknown == ()


def test_spec_case_unknown_tokens_are_reported_not_guessed() -> None:
    parsed = parse("some-weird-xyz:abc")
    assert parsed.unknown == ("some-weird-xyz", "abc")
    assert all(token.status == "unknown" for token in parsed.tokens)
    assert parsed.family is None
    assert parsed.size is None
    assert parsed.tag == "abc"
    assert parsed.tag_is_default is False


# --------------------------------------------------------------- structure ----


@pytest.mark.parametrize(
    ("name", "kinds"),
    [
        ("llama3", ("ollama",)),
        ("qwen3.5:4b", ("ollama",)),
        ("someone/model:tag", ("ollama",)),
        ("registry.ollama.ai/library/llama3.2:3b", ("ollama",)),
        ("hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M", ("ollama",)),
        ("Llama-3-8B-Instruct.Q4_K_M.gguf", ("file",)),
        ("model.safetensors", ("file",)),
        ("https://huggingface.co/google/gemma-3-4b-it", ("huggingface",)),
        # These fit two formats and nothing in the name settles which.
        ("bartowski/Qwen3-30B-A3B-Instruct-GGUF", ("huggingface", "ollama")),
        ("hf.co/google/gemma-3-4b-it", ("huggingface", "ollama")),
    ],
)
def test_kinds(name: str, kinds: tuple[str, ...]) -> None:
    assert parse(name).kinds == kinds


def test_org_repo_without_a_tag_keeps_both_readings() -> None:
    parsed = parse("bartowski/Qwen3-30B-A3B-Instruct-GGUF")
    assert parsed.kinds == ("huggingface", "ollama")
    assert parsed.is_ambiguous
    assert parsed.namespace == "bartowski"
    assert parsed.model == "Qwen3-30B-A3B-Instruct-GGUF"
    # The tag only exists under the Ollama reading, so it is flagged as a default
    # and "latest" is not presented as a fact about the name.
    assert (parsed.tag, parsed.tag_is_default) == ("latest", True)
    assert parsed.build is None


def test_ollama_name_with_host_and_namespace() -> None:
    parsed = parse("registry.ollama.ai/library/llama3.2:3b-instruct-fp16")
    assert parsed.host == "registry.ollama.ai"
    assert parsed.namespace == "library"
    assert parsed.model == "llama3.2"
    assert parsed.tag == "3b-instruct-fp16"
    assert parsed.org == "library"
    assert (parsed.family, parsed.version, parsed.size) == ("llama", "3.2", "3B")
    assert (parsed.tuning, parsed.quantization) == ("instruct", "fp16")
    assert [token.part for token in parsed.tokens[:2]] == ["host", "org"]


def test_ollama_name_pulled_from_hugging_face() -> None:
    parsed = parse("hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M")
    assert parsed.host == "hf.co"
    assert parsed.org == "bartowski"
    assert parsed.quantization == "Q4_K_M"
    assert parsed.tag_is_default is False


def test_default_tag_is_implied_for_a_bare_ollama_name() -> None:
    parsed = parse("llama3")
    assert parsed.build == "latest"
    assert [(item.reading.category, item.reading.value) for item in parsed.implied] == [
        ("build", "latest"),
    ]
    assert "no tag" in parsed.implied[0].reason


def test_explicit_tag_is_not_a_default() -> None:
    parsed = parse("llama3:latest")
    assert (parsed.tag, parsed.tag_is_default) == ("latest", False)
    assert parsed.build == "latest"
    assert [token.text for token in parsed.tokens] == ["llama", "3", "latest"]


def test_file_name_with_a_path() -> None:
    for name in (
        "/models/gemma-3-4b-it-Q4_K_M.gguf",
        "C:\\models\\gemma-3-4b-it-Q4_K_M.gguf",
        "models/sub/gemma-3-4b-it-Q4_K_M.gguf",
    ):
        parsed = parse(name)
        assert parsed.kinds == ("file",)
        assert parsed.filename == "gemma-3-4b-it-Q4_K_M.gguf"
        assert parsed.model == "gemma-3-4b-it-Q4_K_M"
        # A folder name is never mistaken for a publisher.
        assert parsed.namespace is None
        assert parsed.org is None
        assert (parsed.family, parsed.size, parsed.format) == ("gemma", "4B", "gguf")
        assert parsed.tag is None


def test_file_name_with_publisher_prefix() -> None:
    parsed = parse("meta-llama_Llama-3.3-70B-Instruct-Q4_K_M.gguf")
    assert parsed.org == "meta-llama"
    assert (parsed.family, parsed.version, parsed.size) == ("llama", "3.3", "70B")
    assert parsed.unknown == ()


def test_hugging_face_url() -> None:
    parsed = parse("https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main?x=1")
    assert parsed.kinds == ("huggingface",)
    assert parsed.host == "huggingface.co"
    assert parsed.namespace == "google"
    assert parsed.model == "gemma-3-4b-it-qat-q4_0-gguf"
    assert parsed.tag is None
    assert parsed.values("quantization") == ("qat", "Q4_0")


def test_digest_is_ignored() -> None:
    parsed = parse("llama3.2@sha256:0123abcd")
    assert parsed.model == "llama3.2"
    assert (parsed.family, parsed.version) == ("llama", "3.2")


@pytest.mark.parametrize("name", ["", "   ", "/", ":tag", "a/:tag", "@"])
def test_no_name_at_all_is_an_error(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        parse(name)


def test_overlong_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="longer than"):
        parse("a" * 513)


@pytest.mark.parametrize(
    "name",
    [
        "---",
        "::",
        "a:b:c",
        "a//b",
        "a/",
        "name with spaces:tag",
        "weird+chars(1)[2]=3,4",
        "\u00c0\u00c9\u00ce-3",
        "\u0130stanbul3:7b",  # lowercasing this changes its length
        "llama3:",
        "x" * 512,
        "-".join(["qwen3"] * 80),
        "\U0001f600:\U0001f600",
        "..gguf",
        "q4_k_m",
    ],
)
def test_odd_input_never_crashes(name: str) -> None:
    parsed = parse(name)
    assert isinstance(parsed, ParsedName)
    json.dumps(parsed.to_dict())


# ------------------------------------------------------------------ tokens ----


def test_unknown_family_is_not_guessed_from_its_shape() -> None:
    parsed = parse("foo3:7b")
    assert parsed.unknown == ("foo3",)
    assert parsed.family is None
    assert parsed.version is None
    assert parsed.size == "7B"


def test_adjacent_unknown_parts_merge_into_one_token() -> None:
    parsed = parse("openhermes-2.5-mistral-7b.Q4_K_M.gguf")
    assert parsed.unknown == ("openhermes-2.5",)
    assert (parsed.family, parsed.size, parsed.quantization) == ("mistral", "7B", "Q4_K_M")


def test_bare_number_is_a_version_only_after_a_family() -> None:
    assert parse("Llama-3-8B").version == "3"
    assert parse("mystery-3-8B").version is None
    assert parse("mystery-3-8B").unknown == ("mystery-3",)


def test_v_prefixed_version_is_recognized_anywhere() -> None:
    parsed = parse("mistralai/Mistral-7B-Instruct-v0.3")
    assert (parsed.family, parsed.size, parsed.tuning, parsed.version) == (
        "mistral",
        "7B",
        "instruct",
        "v0.3",
    )


def test_dashed_version() -> None:
    parsed = parse("claude-3-5-sonnet-20241022")
    assert (parsed.family, parsed.version, parsed.tier) == ("claude", "3.5", "sonnet")
    assert parsed.values("build") == ("2024-10-22", "latest")


@pytest.mark.parametrize(
    ("name", "texts"),
    [
        ("Llama-3-8B-Instruct.Q4_K_M.gguf", ["Llama", "3", "8B", "Instruct", "Q4_K_M", "gguf"]),
        ("gpt-oss:20b", ["gpt-oss", "20b"]),
        ("mistral-small3.2:24b", ["mistral-small", "3.2", "24b"]),
        ("deepseek-r1:7b", ["deepseek-r1", "7b"]),
        ("Qwen3-Coder", ["Qwen", "3", "Coder"]),
        ("gemma3n:e2b", ["gemma", "3n", "e2b"]),
        ("nomic-embed-text:v1.5", ["nomic-embed-text", "v1.5"]),
        ("llama-guard3:1b", ["llama-guard", "3", "1b"]),
    ],
)
def test_token_boundaries(name: str, texts: list[str]) -> None:
    assert [token.text for token in parse(name).tokens] == texts


def test_multi_part_family_beats_its_parts() -> None:
    parsed = parse("mistral-small3.2:24b-instruct-2506-q4_K_M")
    assert (parsed.family, parsed.version, parsed.size) == ("mistral-small", "3.2", "24B")
    assert (parsed.build, parsed.quantization) == ("2025-06", "Q4_K_M")


def test_two_families_in_one_name() -> None:
    parsed = parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assert parsed.values("family") == ("deepseek-r1", "qwen")
    assert parsed.tuning == "distill"
    assert parsed.org == "deepseek-ai"


def test_several_tokens_in_one_category() -> None:
    parsed = parse("mlx-community/gemma-3-27b-it-qat-4bit")
    assert parsed.org == "mlx-community"
    assert parsed.values("quantization") == ("qat", "4bit")
    assert parsed.quantization == "qat"


def test_values_are_normalized_but_text_is_kept_as_written() -> None:
    parsed = parse("bartowski/Qwen3-30B-A3B-Instruct-GGUF")
    by_text = {token.text: token for token in parsed.tokens}
    assert by_text["Qwen"].reading is not None
    assert by_text["Qwen"].reading.value == "qwen"
    assert by_text["30B"].reading is not None
    assert by_text["30B"].reading.label == "30B parameters"
    assert by_text["GGUF"].category == "format"


def test_context_and_date_tokens() -> None:
    parsed = parse("microsoft/Phi-3-mini-128k-instruct")
    assert (parsed.family, parsed.version, parsed.tier, parsed.context) == (
        "phi",
        "3",
        "mini",
        "128k",
    )
    assert parse("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF").build == "2025-07"


# --------------------------------------------------------------- ambiguity ----


def test_ambiguous_date_stamp_shows_both_readings() -> None:
    parsed = parse("qwen-max-0725")
    (token,) = parsed.ambiguous
    assert token.text == "0725"
    assert token.status == "ambiguous"
    assert token.reading is None
    assert {reading.value for reading in token.readings} == {"07-25", "2025-07"}
    assert parsed.is_ambiguous
    # An ambiguous token never fills a field; only the implied default tag does.
    assert parsed.values("build") == ("latest",)


def test_ambiguous_word_shows_both_readings() -> None:
    parsed = parse("bert-base-uncased")
    (token,) = parsed.ambiguous
    assert token.text == "base"
    assert {reading.category for reading in token.readings} == {"tier", "tuning"}
    assert parsed.tier is None
    assert parsed.tuning is None
    assert parsed.unknown == ("bert", "uncased")


def test_unambiguous_date_stamp() -> None:
    parsed = parse("deepseek-r1-0528")
    assert parsed.ambiguous == ()
    assert parsed.build == "05-28"


# ----------------------------------------------------------------- implied ----


@pytest.mark.parametrize(
    "name",
    ["qwen3.5:4b", "qwen-max-0725", "deepseek/deepseek-v4.1-flash", "llama3", "model.gguf"],
)
def test_hosting_is_not_assumed_when_the_name_is_silent(name: str) -> None:
    parsed = parse(name)
    assert parsed.hosting is None
    assert parsed.values("hosting") == ()
    assert all(item.reading.category != "hosting" for item in parsed.implied)


def test_cloud_tag_says_where_the_model_runs() -> None:
    parsed = parse("gpt-oss:120b-cloud")
    assert parsed.hosting == "cloud"
    assert parsed.implied == ()


def test_token_can_imply_another_category() -> None:
    parsed = parse("Qwen3-30B-A3B")
    assert parsed.architecture == "moe"
    implied = {item.reading.category: item for item in parsed.implied}
    assert implied["architecture"].reading.label == "mixture of experts"
    assert implied["architecture"].reason == "implied by 'A3B' in the name"


def test_hosted_family_implies_cloud() -> None:
    parsed = parse("claude-3-5-sonnet-20241022")
    assert parsed.hosting == "cloud"
    assert "local" not in parsed.values("hosting")


def test_rival_families_imply_nothing() -> None:
    parsed = parse("Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2")
    assert parsed.values("family") == ("qwen", "claude")
    assert parsed.hosting is None
    assert parsed.unknown == ()


def test_architecture_is_not_claimed_without_evidence() -> None:
    assert parse("qwen3.5:4b").architecture is None


def test_explicit_token_wins_over_an_implied_one() -> None:
    parsed = parse("Mixtral-8x7B-MoE")
    assert parsed.values("architecture") == ("moe",)
    assert all(item.reading.category != "architecture" for item in parsed.implied)


# ------------------------------------------------------------------ output ----


def test_to_dict_marks_everything_from_the_name() -> None:
    data = parse("gemma4:e2b-it-qat").to_dict()
    assert data["input"] == "gemma4:e2b-it-qat"
    assert data["kinds"] == ["ollama"]
    assert data["parts"] == {
        "host": None,
        "namespace": None,
        "model": "gemma4",
        "tag": "e2b-it-qat",
        "tag_is_default": False,
        "filename": None,
    }
    assert [token["text"] for token in data["tokens"]] == ["gemma", "4", "e2b", "it", "qat"]
    assert all(token["source"] == "name" for token in data["tokens"])
    assert all(item["source"] == "name" for item in data["implied"])
    assert data["tokens"][0]["readings"] == [
        {
            "category": "family",
            "value": "gemma",
            "label": "Gemma",
            "explanation": "Google's family of open-weight models, built from the same research "
            "as Gemini.",
            "maker": "Google",
        }
    ]
    assert data["tokens"][2]["readings"][0]["label"] == "effective 2B parameters"
    json.dumps(data)


def test_to_dict_keeps_every_reading_of_an_ambiguous_token() -> None:
    data = parse("qwen-max-0725").to_dict()
    token = data["tokens"][-1]
    assert token["status"] == "ambiguous"
    assert len(token["readings"]) == 2
    unknown = parse("some-weird-xyz:abc").to_dict()["tokens"]
    assert [(token["status"], token["readings"]) for token in unknown] == [
        ("unknown", []),
        ("unknown", []),
    ]


# ------------------------------------------------------------ family roles ----


def test_family_after_distill_is_the_model() -> None:
    parsed = parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assert parsed.family_roles == {"deepseek-r1": "source", "qwen": "model"}
    assert parsed.family_roles_reason == (
        "the family right after 'Distill' is the one that was trained"
    )


def test_hosted_only_family_is_a_source() -> None:
    parsed = parse("Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2")
    assert parsed.family_roles == {"qwen": "model", "claude": "source"}
    assert parsed.family_roles_reason == "Claude is only offered as a hosted service"
    assert parsed.to_dict()["family_roles"] == {
        "roles": {"qwen": "model", "claude": "source"},
        "reason": "Claude is only offered as a hosted service",
    }


@pytest.mark.parametrize(
    "name",
    [
        "qwen3.5:4b",  # one family
        "claude-3-5-sonnet-20241022",  # one hosted family
        "lmms-lab/llava-onevision-qwen2-7b",  # two open families, nothing says which
        "some-weird-xyz:abc",  # no family
    ],
)
def test_family_roles_are_not_guessed(name: str) -> None:
    parsed = parse(name)
    assert parsed.family_roles == {}
    assert parsed.family_roles_reason is None


def test_implied_facts_are_labelled_as_implied_in_json() -> None:
    item = parse("Qwen3-30B-A3B").to_dict()["implied"][0]
    assert item["source"] == "implied"
    assert item["reason"] == "implied by 'A3B' in the name"
