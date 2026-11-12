# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A name that says nothing about hosting no longer gets a "Runs locally" row. Where a model runs is shown only when the name states it or a lookup confirms it; a model installed in local Ollama is reported as local by that lookup.
- The `latest` tag Ollama would fill in is no longer shown as a Build row read from the name. It is a note below the table, since a bare name may not be an Ollama name at all.
- README links work on PyPI.

### Changed

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
