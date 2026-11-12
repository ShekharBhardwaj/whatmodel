# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `whatmodel compare <a> <b>...` puts names side by side and marks the fields that differ.
- Names with two families say which is the model and which is a source, when the name settles it (`parsed.family_roles`). Otherwise a note says the name leaves it open.
- About 50 new tokens found by measuring real names: families such as Orca Mini, SQLCoder, OLMo, InternVL, Florence and FLUX; compression formats such as FP8, NVFP4, MXFP8, bitsandbytes and `w4a16`; tunings such as `text`, `python`, `ocr`, `tts` and `asr`.
- `whatmodel -` reads names from standard input, so `ollama list | whatmodel -` works.
- `--brief` prints one row per model.
- Hosted API model IDs (`gpt-4o-2024-08-06`) and OpenRouter IDs (`anthropic/claude-3.5-sonnet:free`) are recognized, and are no longer read as Ollama names.
- Hugging Face answers are cached on disk for a day. `--no-cache` turns it off.
- Unknown tokens come with a link that opens a prefilled GitHub issue.
- `docs/index.html` runs the name reader in the browser.
- `scripts/coverage.py` measures how much of a list of real names the dictionary understands and lists the most common unknown tokens.

### Fixed

- A name that says nothing about hosting no longer gets a "Runs locally" row. Where a model runs is shown only when the name states it or a lookup confirms it; a model installed in local Ollama is reported as local by that lookup.
- The `latest` tag Ollama would fill in is no longer shown as a Build row read from the name. It is a note below the table, since a bare name may not be an Ollama name at all.
- README links work on PyPI.

### Changed

- Rows that a token implies rather than states now show a Source of `name (implied)`, and `implied` in JSON (was `name`).

- `whatmodel tokens` prints grid tables.

### Added

- `whatmodel <name>...` explains Ollama names, Hugging Face repos and GGUF file names in plain English.
- Layer 1 reads the name offline: family and version, publisher, size, effective size, active parameters, tier, tuning, quantization, format, context, build stamps and hosting.
- Unknown tokens are reported as unknown, and ambiguous tokens and name formats show every reading.
- Layer 2 looks up quantization, context length, parameter count, architecture, tool, vision and thinking support, license and file size from the local Ollama API and the Hugging Face Hub API.
- `whatmodel list` explains every model installed in local Ollama.
- `whatmodel tokens` prints the token dictionary.
- Flags `--json`, `--offline`, `--ollama-url`, `--verbose` and `--version`.
- Library API: `parse()` and `explain()`.
- Token dictionary in `whatmodel/tokens.yaml`, the only place token meanings live.
- CI on Python 3.10 to 3.13 across Linux, macOS and Windows.
- Release workflow that publishes to PyPI with Trusted Publishing when a GitHub Release is published.

[Unreleased]: https://github.com/ShekharBhardwaj/whatmodel/commits/main
