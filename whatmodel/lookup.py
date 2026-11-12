"""Layer 2: look up the facts a model name cannot show.

Two sources are tried in order and merged, first answer wins per fact:

1. The local Ollama API: ``POST /api/show`` and ``GET /api/tags``.
2. The Hugging Face Hub API: ``GET /api/models/{org}/{repo}``.

Every lookup is optional. A source that is offline, slow, or does not know the
model is skipped quietly and reported only as a status, never as an error.

Endpoint and field names were checked against the Ollama API reference
(docs/openapi.yaml in ollama/ollama), a running Ollama 0.40, the official
``huggingface_hub`` client and live Hub responses, in November 2026.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from whatmodel import cache
from whatmodel.dictionary import get_dictionary
from whatmodel.parser import (
    HUGGINGFACE_HOSTS,
    KIND_FILE,
    KIND_HUGGINGFACE,
    KIND_OLLAMA,
    ParsedName,
    parse,
)

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
HUGGINGFACE_URL = "https://huggingface.co"

#: Seconds. Ollama is local, so it gets less patience than the Hub.
OLLAMA_TIMEOUT = 2.0
HUGGINGFACE_TIMEOUT = 5.0

SOURCE_OLLAMA = "ollama"
SOURCE_HUGGINGFACE = "huggingface"

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"  # --offline
STATUS_NOT_APPLICABLE = "not_applicable"  # this kind of name cannot be looked up there
STATUS_NOT_FOUND = "not_found"
STATUS_UNREACHABLE = "unreachable"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

_LICENSE_MAX = 60


@dataclass(frozen=True)
class Fact:
    """One looked-up fact, stamped with the source it came from."""

    key: str
    value: Any
    label: str
    source: str
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "key": self.key,
            "value": self.value,
            "label": self.label,
            "source": self.source,
        }
        if self.explanation is not None:
            data["explanation"] = self.explanation
        return data


@dataclass(frozen=True)
class ModelInfo:
    """Layer 1 plus whatever Layer 2 could find."""

    parsed: ParsedName
    facts: tuple[Fact, ...] = ()
    #: Source name -> status, for every source, including the ones skipped.
    lookups: Mapping[str, str] = field(default_factory=dict)
    #: Sources whose facts came from the on-disk cache rather than a fresh request.
    cached: tuple[str, ...] = ()

    def fact(self, key: str) -> Fact | None:
        for fact in self.facts:
            if fact.key == key:
                return fact
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.parsed.to_dict(),
            "facts": [fact.to_dict() for fact in self.facts],
            "lookups": dict(self.lookups),
            "cached": list(self.cached),
        }


class OllamaUnavailableError(RuntimeError):
    """Raised by :func:`list_installed` when the local Ollama cannot be listed."""


# ------------------------------------------------------------- formatting ----


def format_count(count: float) -> str:
    """Format a parameter count the way model names do: 4.6B, 268M."""
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= limit:
            scaled = count / limit
            text = f"{scaled:.0f}" if scaled >= 100 else f"{scaled:.1f}"
            return f"{text.removesuffix('.0')}{suffix}"
    return str(int(count))


def format_context(tokens: int) -> str:
    """Format a context length: 131072 -> 128k tokens, 200000 -> 200k tokens."""
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}k tokens"
    if tokens >= 1000:
        return f"{round(tokens / 1000)}k tokens"
    return f"{tokens} tokens"


def format_bytes(size: float) -> str:
    """Format a file size in decimal units, as Ollama and the Hub do."""
    for limit, suffix in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if size >= limit:
            return f"{size / limit:.1f} {suffix}"
    return f"{int(size)} B"


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _described(key: str, raw: str, category: str, source: str) -> Fact:
    """Build a fact whose value is explained by the token dictionary when it can be."""
    reading = get_dictionary().describe(raw, category=category)
    if reading is None:
        return Fact(key, raw, raw, source)
    label = reading.label if raw.lower() in reading.label.lower() else f"{raw} - {reading.label}"
    return Fact(key, raw, label, source, reading.explanation)


def _short_license(text: str) -> str:
    """Reduce a full license text to its title, keeping a version line if one follows."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    title = lines[0]
    if len(lines) > 1 and lines[1].lower().startswith("version "):
        title = f"{title}, {lines[1].split(',')[0]}"
    return title if len(title) <= _LICENSE_MAX else title[: _LICENSE_MAX - 1] + "…"


# -------------------------------------------------------------- transport ----


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    timeout: float,
    json: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Make one request and fold every failure into a status. Never raises."""
    import httpx

    try:
        response = client.request(method, url, json=json, params=params, timeout=timeout)
    except httpx.TimeoutException:
        return STATUS_TIMEOUT, {}
    except httpx.TransportError:
        return STATUS_UNREACHABLE, {}
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return STATUS_ERROR, {}
    # The Hub answers 401 rather than 404 for a repo that does not exist.
    if response.status_code in (401, 403, 404):
        return STATUS_NOT_FOUND, {}
    if response.status_code != 200:
        return STATUS_ERROR, {}
    try:
        body = response.json()
    except ValueError:
        return STATUS_ERROR, {}
    if not isinstance(body, dict):
        return STATUS_ERROR, {}
    return STATUS_OK, body


def _new_client() -> httpx.Client:
    import httpx

    from whatmodel import __version__

    return httpx.Client(
        headers={"User-Agent": f"whatmodel/{__version__}"},
        follow_redirects=True,
    )


# ------------------------------------------------------------------ ollama ----


def _ollama_name(parsed: ParsedName) -> str | None:
    """The name to ask Ollama about, or ``None`` if Ollama could not know it."""
    if KIND_OLLAMA not in parsed.kinds:
        return None
    reference = parsed.model if parsed.tag is None else f"{parsed.model}:{parsed.tag}"
    lead = [part for part in (parsed.host, parsed.namespace) if part]
    return "/".join([*lead, reference])


def _same_model(listed: object, wanted: str) -> bool:
    if not isinstance(listed, str):
        return False
    return listed == wanted or listed == f"{wanted}:latest" or f"{listed}:latest" == wanted


def ollama_facts(show: Mapping[str, Any], summary: Mapping[str, Any] | None = None) -> list[Fact]:
    """Turn an ``/api/show`` body (and optional ``/api/tags`` entry) into facts."""
    source = SOURCE_OLLAMA
    facts: list[Fact] = []
    details = show.get("details") if isinstance(show.get("details"), dict) else {}
    info = show.get("model_info") if isinstance(show.get("model_info"), dict) else {}

    quantization = _text(details.get("quantization_level"))
    if quantization is not None:
        facts.append(_described("quantization", quantization, "quantization", source))

    architecture_name = _text(info.get("general.architecture"))
    context = _positive_int(info.get(f"{architecture_name}.context_length"))
    if context is None:
        context = next(
            (
                found
                for key, value in info.items()
                if key.endswith(".context_length") and (found := _positive_int(value))
            ),
            None,
        )
    if context is not None:
        facts.append(Fact("context_length", context, format_context(context), source))

    parameters = _positive_int(info.get("general.parameter_count"))
    size_label = _text(details.get("parameter_size"))
    if parameters is not None:
        facts.append(Fact("parameter_count", parameters, format_count(parameters), source))
    elif size_label is not None:
        facts.append(Fact("parameter_count", size_label, size_label, source))

    # GGUF metadata carries "<arch>.expert_count" only for mixture-of-experts
    # models. "Dense" is only claimed when the full GGUF metadata is present.
    experts = [_positive_int(v) for k, v in info.items() if k.endswith(".expert_count")]
    if any(experts):
        facts.append(_described("architecture", "moe", "architecture", source))
    elif details.get("format") == "gguf" and architecture_name is not None:
        facts.append(_described("architecture", "dense", "architecture", source))

    capabilities = show.get("capabilities")
    if isinstance(capabilities, list):
        for key in ("tools", "vision", "thinking"):
            flag = key in capabilities
            facts.append(Fact(key, flag, _yes_no(flag), source))

    license_name = _text(info.get("general.license")) or _short_license(
        _text(show.get("license")) or ""
    )
    if license_name:
        facts.append(Fact("license", license_name, license_name, source))

    summary = summary or {}
    size = _positive_int(summary.get("size"))
    if size is not None:
        facts.append(Fact("file_size", size, format_bytes(size), source))
    remote = _text(show.get("remote_host")) or _text(summary.get("remote_host"))
    if remote is not None:
        facts.append(Fact("hosting", "cloud", f"remotely, on {remote}", source))
    elif size is not None:
        # The weights are on this machine's disk, so this is observed, not assumed.
        facts.append(Fact("hosting", "local", "locally, installed in Ollama here", source))
    return facts


def _lookup_ollama(
    parsed: ParsedName, client: httpx.Client, base_url: str, timeout: float
) -> tuple[str, list[Fact]]:
    name = _ollama_name(parsed)
    if name is None:
        return STATUS_NOT_APPLICABLE, []
    base = base_url.rstrip("/")
    status, show = _request(
        client, "POST", f"{base}/api/show", json={"model": name}, timeout=timeout
    )
    if status != STATUS_OK:
        return status, []
    # File size and the remote-host marker live in the model list, not in "show".
    _, listing = _request(client, "GET", f"{base}/api/tags", timeout=timeout)
    models = listing.get("models") if isinstance(listing.get("models"), list) else []
    summary = next(
        (m for m in models if isinstance(m, dict) and _same_model(m.get("name"), name)), None
    )
    return STATUS_OK, ollama_facts(show, summary)


def list_installed(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    *,
    timeout: float = OLLAMA_TIMEOUT,
    client: httpx.Client | None = None,
) -> list[str]:
    """Return the names of the models installed in the local Ollama.

    Unlike a lookup, this cannot be skipped quietly: without the list there is
    nothing to explain, so failure raises :class:`OllamaUnavailableError`.
    """
    owned = client is None
    active = client or _new_client()
    try:
        status, body = _request(
            active, "GET", f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout
        )
    finally:
        if owned:
            active.close()
    if status != STATUS_OK:
        raise OllamaUnavailableError(f"could not list models from {ollama_url} ({status})")
    models = body.get("models") if isinstance(body.get("models"), list) else []
    names = [m.get("name") for m in models if isinstance(m, dict)]
    return [name for name in names if isinstance(name, str) and name]


# ------------------------------------------------------------ hugging face ----


def _huggingface_repo(parsed: ParsedName) -> str | None:
    """The ``org/repo`` to ask the Hub about, or ``None`` if the name has none."""
    if parsed.namespace is None or KIND_FILE in parsed.kinds:
        return None
    on_hub = parsed.host is not None and parsed.host.lower() in HUGGINGFACE_HOSTS
    if KIND_HUGGINGFACE in parsed.kinds or on_hub:
        return f"{parsed.namespace}/{parsed.model}"
    return None


def _chat_template(body: Mapping[str, Any]) -> str | None:
    gguf = body.get("gguf") if isinstance(body.get("gguf"), dict) else {}
    config = body.get("config") if isinstance(body.get("config"), dict) else {}
    tokenizer = config.get("tokenizer_config")
    tokenizer = tokenizer if isinstance(tokenizer, dict) else {}
    for candidate in (gguf.get("chat_template"), tokenizer.get("chat_template")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _gguf_files(body: Mapping[str, Any]) -> list[tuple[str, int]]:
    siblings = body.get("siblings") if isinstance(body.get("siblings"), list) else []
    files: list[tuple[str, int]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name, size = item.get("rfilename"), _positive_int(item.get("size"))
        if isinstance(name, str) and name.lower().endswith(".gguf") and size is not None:
            files.append((name, size))
    return files


def huggingface_facts(
    body: Mapping[str, Any], wanted_quantization: str | None = None
) -> list[Fact]:
    """Turn a Hub ``/api/models/{org}/{repo}`` body into facts."""
    source = SOURCE_HUGGINGFACE
    facts: list[Fact] = []
    gguf = body.get("gguf") if isinstance(body.get("gguf"), dict) else {}
    safetensors = body.get("safetensors") if isinstance(body.get("safetensors"), dict) else {}
    card = body.get("cardData") if isinstance(body.get("cardData"), dict) else {}
    tags = body.get("tags") if isinstance(body.get("tags"), list) else []

    # A safetensors repo states its weight type, such as BF16, per parameter count.
    dtypes = safetensors.get("parameters")
    if isinstance(dtypes, dict) and dtypes:
        main = max(dtypes, key=lambda name: _positive_int(dtypes[name]) or 0)
        if isinstance(main, str):
            facts.append(_described("quantization", main, "quantization", source))

    context = _positive_int(gguf.get("context_length"))
    if context is not None:
        facts.append(Fact("context_length", context, format_context(context), source))

    parameters = _positive_int(gguf.get("total")) or _positive_int(safetensors.get("total"))
    if parameters is not None:
        facts.append(Fact("parameter_count", parameters, format_count(parameters), source))

    template = _chat_template(body)
    if template is not None:
        flag = "tools" in template
        facts.append(
            Fact("tools", flag, f"{_yes_no(flag)} (judged from the chat template)", source)
        )
    if body.get("pipeline_tag") == "image-text-to-text":
        facts.append(Fact("vision", True, "yes", source))

    license_value = card.get("license")
    if isinstance(license_value, list):
        license_value = ", ".join(str(item) for item in license_value)
    license_name = _text(license_value) or next(
        (
            tag.split(":", 1)[1]
            for tag in tags
            if isinstance(tag, str) and tag.startswith("license:")
        ),
        None,
    )
    if license_name:
        facts.append(Fact("license", license_name, license_name, source))

    files = _gguf_files(body)
    wanted = (wanted_quantization or "").lower()
    chosen = [f for f in files if wanted and wanted in f[0].lower()]
    if len(chosen) == 1:
        name, size = chosen[0]
        facts.append(Fact("file_size", size, f"{format_bytes(size)} ({name})", source))
    elif files:
        sizes = sorted(size for _, size in files)
        label = (
            format_bytes(sizes[0])
            if len(sizes) == 1
            else f"{format_bytes(sizes[0])} to {format_bytes(sizes[-1])}, {len(sizes)} GGUF files"
        )
        value = {"min": sizes[0], "max": sizes[-1], "files": len(sizes)}
        facts.append(Fact("file_size", value, label, source))
    return facts


def _lookup_huggingface(
    parsed: ParsedName, client: httpx.Client, timeout: float, cache_dir: Path | None = None
) -> tuple[str, list[Fact], bool]:
    """Return the status, the facts, and whether they came from the cache."""
    repo = _huggingface_repo(parsed)
    if repo is None:
        return STATUS_NOT_APPLICABLE, [], False
    if cache_dir is not None:
        stored = cache.read(cache_dir, repo)
        if stored is not None:
            return STATUS_OK, huggingface_facts(stored, parsed.quantization), True
    status, body = _request(
        client,
        "GET",
        f"{HUGGINGFACE_URL}/api/models/{repo}",
        params={"blobs": "true"},
        timeout=timeout,
    )
    if status != STATUS_OK:
        return status, [], False
    if cache_dir is not None:
        cache.write(cache_dir, repo, body)
    return STATUS_OK, huggingface_facts(body, parsed.quantization), False


# ------------------------------------------------------------------- merge ----


def _merge(groups: Sequence[Sequence[Fact]]) -> tuple[Fact, ...]:
    """Keep the first fact seen for each key; sources are given in priority order."""
    merged: dict[str, Fact] = {}
    for facts in groups:
        for fact in facts:
            merged.setdefault(fact.key, fact)
    return tuple(merged.values())


def explain(
    name: str,
    *,
    offline: bool = False,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float | None = None,
    client: httpx.Client | None = None,
    cache_dir: Path | None = None,
) -> ModelInfo:
    """Explain a model name: Layer 1 from the name, then Layer 2 from lookups.

    ``offline=True`` skips Layer 2 completely and makes no request. ``timeout``
    overrides the per-source defaults. ``client`` lets a caller share or mock
    the HTTP client. ``cache_dir`` turns on a one-day on-disk cache of Hugging
    Face answers; without it nothing is read from or written to disk. Lookup
    failures never raise.
    """
    parsed = parse(name)
    if offline:
        skipped = {SOURCE_OLLAMA: STATUS_SKIPPED, SOURCE_HUGGINGFACE: STATUS_SKIPPED}
        return ModelInfo(parsed, (), skipped)

    wants_ollama = _ollama_name(parsed) is not None
    wants_hub = _huggingface_repo(parsed) is not None
    if not (wants_ollama or wants_hub):
        nothing = {SOURCE_OLLAMA: STATUS_NOT_APPLICABLE, SOURCE_HUGGINGFACE: STATUS_NOT_APPLICABLE}
        return ModelInfo(parsed, (), nothing)

    owned = client is None
    active = client or _new_client()
    try:
        ollama_status, from_ollama = _lookup_ollama(
            parsed, active, ollama_url, OLLAMA_TIMEOUT if timeout is None else timeout
        )
        hub_status, from_hub, hub_cached = _lookup_huggingface(
            parsed, active, HUGGINGFACE_TIMEOUT if timeout is None else timeout, cache_dir
        )
    finally:
        if owned:
            active.close()
    lookups = {SOURCE_OLLAMA: ollama_status, SOURCE_HUGGINGFACE: hub_status}
    cached = (SOURCE_HUGGINGFACE,) if hub_cached else ()
    return ModelInfo(parsed, _merge([from_ollama, from_hub]), lookups, cached)
