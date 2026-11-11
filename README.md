# whatmodel

Explain LLM model names in plain English: give it `gemma4:e2b-it-qat` and it tells you what each part means, plus the facts the name leaves out.

It reads names from Ollama, Hugging Face and GGUF files. There are no LLM calls: every explanation comes from a plain data file and from model metadata.

## Install

```bash
uv tool install whatmodel
```

Or run it once without installing:

```bash
uvx whatmodel qwen3.5:4b
```

`pip install whatmodel` works too. Python 3.10 or newer.

## Examples

The outputs below are pasted from real runs.

### An Ollama name

```console
$ whatmodel gemma4:e2b-it-qat

  gemma4:e2b-it-qat

  Family       Gemma 4 (Google)
  Size         effective 2B parameters
  Tuning       instruction tuned (ready for chat)
  Compression  quantization-aware trained
  Runs         locally

  Not in the name  (from local Ollama)
  Parameters    4.6B
  Compression   4-bit (Q4_0)
  Architecture  dense
  Context       128k tokens
  Tools         yes
  Vision        yes
  Thinking      yes
  License       Apache License, Version 2.0
  File size     4.3 GB
```

The top block is read from the name alone. The second block is looked up, here from the Ollama running on the same machine.

### A Hugging Face repo

```console
$ whatmodel bartowski/Qwen_Qwen3-30B-A3B-GGUF

  bartowski/Qwen_Qwen3-30B-A3B-GGUF
  Could be a Hugging Face repo or an Ollama model (tag latest); the name
  alone does not say which.

  Publisher     bartowski (community GGUF conversions)
  Family        Qwen 3 (Alibaba)
  Size          30B parameters, 3B active per token
  Architecture  mixture of experts
  Format        GGUF (for llama.cpp, Ollama and LM Studio)
  Runs          locally

  Not in the name  (from Hugging Face)
  Parameters  30.5B
  Context     32k tokens
  Tools       yes (judged from the chat template)
  License     apache-2.0
  File size   8.1 GB to 39.7 GB, 28 GGUF files
```

### A GGUF file name

```console
$ whatmodel Llama-3-8B-Instruct.Q4_K_M.gguf

  Llama-3-8B-Instruct.Q4_K_M.gguf

  Family       Llama 3 (Meta)
  Size         8B parameters
  Tuning       instruction tuned (ready for chat)
  Compression  4-bit (Q4_K_M, the balanced variant)
  Format       GGUF (for llama.cpp, Ollama and LM Studio)
  Runs         locally
```

A file name has nothing to look up, so only the first block appears.

## Usage

```text
whatmodel <name>              explain one name
whatmodel <name> <name>...    explain several
whatmodel list                explain every model installed in local Ollama
whatmodel tokens              print the token dictionary
```

| Flag | What it does |
|---|---|
| `--json` | Print JSON. One name gives an object, several names give a list. |
| `--offline` | Read the name only and make no network requests. |
| `--ollama-url URL` | Where Ollama listens. Defaults to `http://localhost:11434`, or `OLLAMA_HOST` if set. |
| `--verbose`, `-v` | Add a one-sentence explanation under each row. |
| `--version` | Print the version. |

Accepted name formats:

- Ollama: `[host/][namespace/]model[:tag]`, for example `qwen3.5:4b` or `hf.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF:Q4_K_M`
- Hugging Face: `org/repo`, or a pasted `https://huggingface.co/org/repo` link
- File names: `Llama-3-8B-Instruct.Q4_K_M.gguf`, with or without a folder in front

A model that happens to be called `list` or `tokens` can be explained with `whatmodel explain list`.

### Unknown and ambiguous tokens

whatmodel never guesses. A token that is not in its dictionary is reported as unknown. A token with two possible meanings is shown with both:

```console
$ whatmodel qwen-max-0725 --offline

  qwen-max-0725

  Family     Qwen (Alibaba)
  Tier       max tier (the most capable option)
  Build      latest (the default build)
  Runs       locally
  Ambiguous  0725: dated 07-25 (month-day) [build]
             or released 2025-07 [build]
```

The same applies to a name such as `org/model` with no tag, which fits both a Hugging Face repo and an Ollama model.

### Where the facts come from

The lookup step tries two sources in order and merges them:

1. The local Ollama API (`/api/show` and `/api/tags`), for models you have installed.
2. The Hugging Face Hub API, for names that contain `org/repo`.

Timeouts are short. A source that is offline, slow, or does not know the model is skipped, and the name is still explained. In JSON every item carries a `source` of `name`, `ollama` or `huggingface`, and `lookups` reports what happened with each source.

## Library

```python
from whatmodel import parse, explain

parsed = parse("qwen3.5:4b")  # the name only, no network
parsed.family, parsed.version, parsed.size  # ('qwen', '3.5', '4B')
parsed.unknown  # tokens the dictionary does not know

info = explain("qwen3.5:4b")  # the name plus lookups
info.facts  # looked-up facts, each with its source
info.to_dict()  # the same data --json prints
```

`explain(name, offline=True)` makes no requests at all.

## What the name cannot tell you

A model name is a label chosen by whoever published it. Even a fully understood name leaves these open, which is what the lookup step is for:

- **The exact compression.** `qat` says how a model was trained, not how many bits it uses. A bare tag such as `:4b` says nothing about compression at all.
- **What `latest` points to.** It is a moving pointer. The size and compression behind it can change.
- **The real parameter count.** `4b` is a rounded label. The model above is called `e2b` and has 4.6 billion parameters.
- **Context length.** Rarely in the name, and the usable value also depends on how the model is run.
- **Capabilities.** Tool calling, vision and thinking are usually not in the name.
- **The license.** Never in the name.
- **Who trained it.** The publisher in `org/repo` is often a third party who converted the model.
- **File size and memory needs.** These depend on compression and format. whatmodel reports file size when a source has it and does not estimate whether a model fits your hardware.

And some things no lookup settles: a name does not tell you how good a model is.

## Adding a token

Every meaning whatmodel knows lives in one file, [`whatmodel/tokens.yaml`](whatmodel/tokens.yaml). No Python is needed to teach it a new token:

```yaml
- pattern: olmo
  category: family
  label: OLMo
  maker: Allen Institute for AI
  explanation: The open language model family from the Allen Institute for AI.
```

Then check it:

```bash
uv run pytest
uv run whatmodel olmo-2:7b --offline
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers patterns, regex entries and the rules entries must follow.

## License

MIT. See [LICENSE](LICENSE).
