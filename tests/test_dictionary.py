"""Tests for tokens.yaml and the code that loads it. No network."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from whatmodel.dictionary import (
    CORE_CATEGORIES,
    Dictionary,
    DictionaryError,
    build_dictionary,
    get_dictionary,
    load_dictionary,
    normalize,
    split_pieces,
)

# Every category the project spec asks the starter set to cover.
SPEC_CATEGORIES = (
    "family",
    "version",
    "org",
    "size",
    "effective_size",
    "active_params",
    "tier",
    "tuning",
    "quantization",
    "format",
    "hosting",
    "build",
)

# Every example token the project spec lists, with the category it must land in.
SPEC_TOKENS = [
    ("4b", "size"),
    ("12b", "size"),
    ("270m", "size"),
    ("e2b", "effective_size"),
    ("e4b", "effective_size"),
    ("a3b", "active_params"),
    ("a22b", "active_params"),
    ("pro", "tier"),
    ("flash", "tier"),
    ("mini", "tier"),
    ("nano", "tier"),
    ("lite", "tier"),
    ("turbo", "tier"),
    ("max", "tier"),
    ("it", "tuning"),
    ("instruct", "tuning"),
    ("chat", "tuning"),
    ("base", "tuning"),
    ("code", "tuning"),
    ("vision", "tuning"),
    ("embed", "tuning"),
    ("thinking", "tuning"),
    ("q4_k_m", "quantization"),
    ("q8_0", "quantization"),
    ("fp16", "quantization"),
    ("bf16", "quantization"),
    ("qat", "quantization"),
    ("awq", "quantization"),
    ("exl2", "quantization"),
    ("4bit", "quantization"),
    ("gguf", "format"),
    ("safetensors", "format"),
    ("mlx", "format"),
    ("onnx", "format"),
    ("cloud", "hosting"),
    ("2507", "build"),
    ("0725", "build"),
]

# Tokens that are meant to have more than one reading. Anything else that
# matches two entries is an accident and fails the test below.
INTENDED_AMBIGUOUS = {"base", "0725", "0126"}


@pytest.fixture(scope="module")
def dictionary() -> Dictionary:
    return get_dictionary()


def _minimal() -> dict[str, Any]:
    """A small valid dictionary that the error tests break one field at a time."""
    categories = {
        name: {"heading": name.title(), "description": "A test."} for name in CORE_CATEGORIES
    }
    return {
        "schema": 1,
        "categories": categories,
        "tokens": [
            {
                "pattern": "llama",
                "category": "family",
                "label": "Llama",
                "explanation": "A test family.",
            }
        ],
    }


def _with_entry(**changes: Any) -> dict[str, Any]:
    data = _minimal()
    data["tokens"][0].update(changes)
    return data


# ---------------------------------------------------------------- the file ----


def test_shipped_dictionary_loads(dictionary: Dictionary) -> None:
    assert dictionary.entries
    assert set(CORE_CATEGORIES) <= set(dictionary.categories)


@pytest.mark.parametrize("category", SPEC_CATEGORIES)
def test_every_spec_category_has_entries(dictionary: Dictionary, category: str) -> None:
    assert category in dictionary.categories
    assert dictionary.entries_in(category), f"no starter entries for '{category}'"


@pytest.mark.parametrize(("token", "category"), SPEC_TOKENS)
def test_spec_tokens_are_known(dictionary: Dictionary, token: str, category: str) -> None:
    readings = dictionary.match(normalize(token), ignore_context=True)
    assert category in {reading.category for reading in readings}


def test_regex_entries_match_their_examples(dictionary: Dictionary) -> None:
    regex_entries = [entry for entry in dictionary.entries if entry.is_regex]
    assert regex_entries
    for entry in regex_entries:
        assert entry.examples, f"{entry.patterns} has no examples"
        for example in entry.examples:
            assert entry.groups(normalize(example)) is not None, (entry.patterns, example)


def test_no_accidental_ambiguity(dictionary: Dictionary) -> None:
    """Two entries may only claim the same token when that is the intent."""
    samples: set[str] = set()
    for entry in dictionary.entries:
        samples.update(entry.literals)
        samples.update(normalize(example) for example in entry.examples)
    surprises = {
        sample: [reading.category for reading in dictionary.match(sample, ignore_context=True)]
        for sample in sorted(samples)
        if len(dictionary.match(sample, ignore_context=True)) > 1
        and sample not in INTENDED_AMBIGUOUS
    }
    assert not surprises


@pytest.mark.parametrize("token", sorted(INTENDED_AMBIGUOUS))
def test_intended_ambiguity_is_kept(dictionary: Dictionary, token: str) -> None:
    assert len(dictionary.match(token, ignore_context=True)) == 2


def test_explanations_are_one_plain_sentence(dictionary: Dictionary) -> None:
    texts = [(entry.patterns, entry.explanation) for entry in dictionary.entries]
    texts += [((category.id,), category.description) for category in dictionary.categories.values()]
    for owner, text in texts:
        assert text.endswith("."), (owner, text)
        assert ". " not in text, (owner, "more than one sentence")
        assert "\n" not in text, owner
        assert len(text) <= 200, (owner, len(text))


def test_labels_are_short(dictionary: Dictionary) -> None:
    for entry in dictionary.entries:
        assert len(entry.label) <= 60, entry.label


def test_implied_tokens_resolve(dictionary: Dictionary) -> None:
    implying = [entry for entry in dictionary.entries if entry.implies]
    assert implying
    for entry in implying:
        for category, token in entry.implies:
            assert dictionary.describe(token, category=category) is not None


def test_category_default(dictionary: Dictionary) -> None:
    default = dictionary.categories["hosting"].default
    assert default is not None
    assert default.label == "locally"
    assert default.value == "local"


# ---------------------------------------------------------------- matching ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Q4_K_M", "q4_k_m"),
        ("gpt-oss", "gpt_oss"),
        ("hf.co", "hf_co"),
        ("qwen3.5", "qwen3.5"),
        ("v0.3.Q4", "v0.3_q4"),
        ("Llama-3-8B-Instruct.Q4_K_M.gguf", "llama_3_8b_instruct_q4_k_m_gguf"),
        ("  spaced   out  ", "spaced_out"),
        ("---", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_split_pieces_keeps_positions() -> None:
    pieces = split_pieces("Qwen3.5-Coder")
    assert [(piece.text, piece.start, piece.end) for piece in pieces] == [
        ("Qwen3.5", 0, 7),
        ("Coder", 8, 13),
    ]


def test_match_is_case_insensitive_and_normalizes_values(dictionary: Dictionary) -> None:
    (reading,) = dictionary.match(normalize("Q4_K_M"))
    assert (reading.category, reading.value) == ("quantization", "Q4_K_M")
    (reading,) = dictionary.match(normalize("12B"))
    assert (reading.category, reading.value) == ("size", "12B")


def test_after_constraint(dictionary: Dictionary) -> None:
    assert dictionary.match("3") == ()
    (reading,) = dictionary.match("3", frozenset({"family"}))
    assert reading.category == "version"


def test_split_fused_needs_a_known_family(dictionary: Dictionary) -> None:
    fused = dictionary.split_fused("qwen3.5")
    assert fused is not None
    assert fused.cut == 4
    assert [reading.value for reading in fused.families] == ["qwen"]
    assert [reading.value for reading in fused.versions] == ["3.5"]
    assert dictionary.split_fused("foo3.5") is None
    assert dictionary.split_fused("qwen") is None
    assert dictionary.split_fused("qwen7b") is None


def test_describe_explains_api_values(dictionary: Dictionary) -> None:
    reading = dictionary.describe("Q4_K_M", category="quantization")
    assert reading is not None
    assert reading.label.startswith("4-bit")
    assert dictionary.describe("not-a-real-quant", category="quantization") is None
    assert dictionary.describe("base") is None  # two readings, so nothing is picked


def test_uppercase_conversion_in_templates() -> None:
    data = _minimal()
    data["tokens"].append(
        {
            "pattern": "iq(?P<bits>[1-4])_xs",
            "regex": True,
            "category": "format",
            "label": "{bits}-bit ({text!u})",
            "value": "{text!u}",
            "explanation": "A test.",
            "examples": ["iq4_xs"],
        }
    )
    (reading,) = build_dictionary(data).match("iq4_xs")
    assert reading.value == "IQ4_XS"
    assert reading.label == "4-bit (IQ4_XS)"


def test_literal_aliases_keep_their_own_value() -> None:
    data = _with_entry(pattern=["llama", "big-llama"])
    built = build_dictionary(data)
    assert built.match("llama")[0].value == "llama"
    assert built.match("big_llama")[0].value == "big-llama"
    assert built.max_parts == 2


# ------------------------------------------------------------ bad dictionaries ----


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "must be a mapping"),
        ({**_minimal(), "extra": 1}, "unknown top-level key"),
        ({**_minimal(), "schema": 2}, "'schema' must be 1"),
        ({**_minimal(), "tokens": []}, "non-empty list"),
        ({**_minimal(), "categories": {}}, "non-empty mapping"),
        (_with_entry(colour="red"), "unknown key"),
        (_with_entry(category="nope"), "unknown category"),
        (_with_entry(label=""), "'label' must be a non-empty string"),
        (_with_entry(pattern=2507), "quote numbers"),
        (_with_entry(pattern="---"), "nothing to match"),
        (_with_entry(regex="yes"), "'regex' must be true or false"),
        (_with_entry(pattern="(", regex=True, examples=["x"]), "invalid regular expression"),
        (_with_entry(pattern="a+", regex=True), "need 'examples'"),
        (_with_entry(pattern="a+", regex=True, examples=["bbb"]), "does not match"),
        (_with_entry(pattern="(?P<text>a)", regex=True, examples=["a"]), "cannot be named 'text'"),
        (_with_entry(label="{nope}"), "uses {nope}"),
        (_with_entry(label="{broken"), "not a valid template"),
        (_with_entry(after="nope"), "'after' names unknown category"),
        (_with_entry(category="format", maker="Someone"), "'maker' is only allowed"),
        (_with_entry(implies={"nope": "x"}), "'implies' names unknown category"),
        (_with_entry(implies={"family": "x"}), "own category"),
        (_with_entry(implies={"format": "missing"}), "must match exactly one entry"),
        (_with_entry(implies="format"), "'implies' must map"),
    ],
)
def test_malformed_dictionaries_are_rejected(data: Any, message: str) -> None:
    with pytest.raises(DictionaryError, match=message):
        build_dictionary(copy.deepcopy(data))


def test_missing_required_key_is_named() -> None:
    data = _minimal()
    del data["tokens"][0]["explanation"]
    with pytest.raises(DictionaryError, match="missing required key 'explanation'"):
        build_dictionary(data)


def test_missing_core_category_is_rejected() -> None:
    data = _minimal()
    del data["categories"]["host"]
    with pytest.raises(DictionaryError, match="missing required id"):
        build_dictionary(data)


def test_duplicate_literal_in_one_category_is_rejected() -> None:
    data = _minimal()
    data["tokens"].append(dict(data["tokens"][0]))
    with pytest.raises(DictionaryError, match="already defined"):
        build_dictionary(data)


def test_bad_default_is_rejected() -> None:
    data = _minimal()
    data["categories"]["host"]["default"] = ["a", "list"]
    with pytest.raises(DictionaryError, match="'default' must be"):
        build_dictionary(data)


def test_load_reports_the_file_name(tmp_path: Any) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("schema: 1\ncategories: [unclosed\n", encoding="utf-8")
    with pytest.raises(DictionaryError, match=r"broken\.yaml: not valid YAML"):
        load_dictionary(broken)

    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("schema: 9\n", encoding="utf-8")
    with pytest.raises(DictionaryError, match=r"wrong\.yaml: 'schema' must be 1"):
        load_dictionary(wrong)
