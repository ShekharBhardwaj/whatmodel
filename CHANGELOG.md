# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Release workflow that publishes to TestPyPI and PyPI with Trusted Publishing when a GitHub Release is published.

[Unreleased]: https://github.com/ShekharBhardwaj/whatmodel/commits/main
