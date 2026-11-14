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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from whatmodel import cache, gguf
from whatmodel.dictionary import get_dictionary
from whatmodel.parser import (
    HUGGINGFACE_HOSTS,
    KIND_API,
    KIND_FILE,
    KIND_HUGGINGFACE,
    KIND_OLLAMA,
    KIND_OPENROUTER,
    ParsedName,
    parse,
)

if TYPE_CHECKING:
    import httpx
    from rich.console import RenderableType

DEFAULT_OLLAMA_URL = "http://localhost:11434"
HUGGINGFACE_URL = "https://huggingface.co"

#: Seconds. Ollama is local, so it gets less patience than the Hub.
OLLAMA_TIMEOUT = 2.0
HUGGINGFACE_TIMEOUT = 5.0
#: OpenRouter's public list of every model it serves. It needs no key.
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_TIMEOUT = 5.0
_OPENROUTER_CACHE_KEY = "openrouter/models"

SOURCE_OLLAMA = "ollama"
SOURCE_HUGGINGFACE = "huggingface"
SOURCE_OPENROUTER = "openrouter"
#: A GGUF file on this machine, read directly.
SOURCE_FILE = "file"

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

    def get(self, key: str, default: Any = None) -> Any:
        """The value of a looked-up fact, or ``default`` if nothing was found for it."""
        found = self.fact(key)
        return default if found is None else found.value

    # What the name says, without going through ``.parsed``.

    @property
    def name(self) -> str:
        return self.parsed.raw

    @property
    def family(self) -> str | None:
        return self.parsed.family

    @property
    def version(self) -> str | None:
        return self.parsed.version

    @property
    def size(self) -> str | None:
        return self.parsed.size

    @property
    def tier(self) -> str | None:
        return self.parsed.tier

    @property
    def tuning(self) -> str | None:
        return self.parsed.tuning

    @property
    def format(self) -> str | None:
        return self.parsed.format

    @property
    def unknown(self) -> tuple[str, ...]:
        return self.parsed.unknown

    # The best answer available: a looked-up fact when there is one, else the name.

    @property
    def quantization(self) -> str | None:
        """The compression, such as ``Q4_K_M``. A looked-up value wins over the name's."""
        found = self.get("quantization")
        return found if isinstance(found, str) else self.parsed.quantization

    @property
    def parameter_count(self) -> int | None:
        """How many parameters, as a number. Looked up if possible, else from the name."""
        found = self.get("parameter_count")
        return found if isinstance(found, int) else self.parsed.parameters

    @property
    def runs(self) -> str | None:
        """``"local"`` or ``"cloud"`` when a lookup or the name establishes it, else ``None``."""
        found = self.get("hosting")
        return found if isinstance(found, str) else self.parsed.hosting

    # Looked-up facts only. ``None`` means nothing was found, never "no".

    @property
    def context_length(self) -> int | None:
        found = self.get("context_length")
        return found if isinstance(found, int) else None

    @property
    def supports_tools(self) -> bool | None:
        return self._flag("tools")

    @property
    def supports_vision(self) -> bool | None:
        return self._flag("vision")

    @property
    def supports_thinking(self) -> bool | None:
        return self._flag("thinking")

    @property
    def license(self) -> str | None:
        found = self.get("license")
        return found if isinstance(found, str) else None

    @property
    def based_on(self) -> tuple[str, ...]:
        """The models this one was made from, when a model card or file header says."""
        found = self.get("base_model")
        models = found.get("models") if isinstance(found, dict) else None
        return tuple(models) if isinstance(models, list) else ()

    def _flag(self, key: str) -> bool | None:
        found = self.get(key)
        return found if isinstance(found, bool) else None

    @property
    def summary(self) -> str:
        """One line: what the name says, then the key looked-up facts."""
        from whatmodel.render import summary

        said, looked_up = summary(self)
        return f"{said} | {looked_up}" if looked_up else said

    def __str__(self) -> str:
        """Plain text, one line per field, with where each fact came from."""
        from whatmodel.render import describe

        return describe(self)

    def __rich__(self) -> RenderableType:
        """Lets ``rich.print(info)`` draw the same table the command line prints."""
        from whatmodel.render import render_info

        return render_info(self)

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
    """Format a context length: 131072 -> 128k tokens, 1000000 -> 1M tokens."""
    for unit in (1024 * 1024, 1_000_000):
        if tokens >= unit and tokens % unit == 0:
            return f"{tokens // unit}M tokens"
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


def explain_many(
    names: Iterable[str],
    *,
    offline: bool = False,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float | None = None,
    client: httpx.Client | None = None,
    cache_dir: Path | None = None,
) -> list[ModelInfo]:
    """Explain several names, reusing one connection for all their lookups.

    Takes the same options as :func:`explain` and returns the results in order.
    """
    owned = client is None and not offline
    active = _new_client() if owned else client
    try:
        return [
            explain(
                name,
                offline=offline,
                ollama_url=ollama_url,
                timeout=timeout,
                client=active,
                cache_dir=cache_dir,
            )
            for name in names
        ]
    finally:
        if owned and active is not None:
            active.close()


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


#: How a repo relates to its base model, as the Hub tags it, in plain words.
_RELATIONS = {
    "quantized": "a compressed copy of it",
    "finetune": "fine-tuned from it",
    "adapter": "an adapter for it",
    "merge": "merged from it",
}


def _base_model(card: Mapping[str, Any], tags: Sequence[Any]) -> Fact | None:
    """The model this repo was made from, as its model card states it."""
    raw = card.get("base_model")
    names = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    names = [name.strip() for name in names if isinstance(name, str) and name.strip()]
    relation = _text(card.get("base_model_relation"))
    for tag in tags:
        # The Hub also writes the relation into a tag: base_model:quantized:org/repo
        parts = tag.split(":", 2) if isinstance(tag, str) else []
        if len(parts) == 3 and parts[0] == "base_model" and parts[1] in _RELATIONS:
            relation = relation or parts[1]
            if parts[2] not in names:
                names.append(parts[2])
    if not names:
        return None
    label = ", ".join(names)
    if relation in _RELATIONS:
        phrase = _RELATIONS[relation]
        if len(names) > 1:
            phrase = phrase.replace("of it", "of these").replace("it", "these")
        label = f"{label} ({phrase})"
    value = {"models": names, "relation": relation if relation in _RELATIONS else None}
    return Fact("base_model", value, label, SOURCE_HUGGINGFACE)


def gguf_facts(metadata: Mapping[str, Any], size: int | None = None) -> list[Fact]:
    """Turn the header of a GGUF file into facts."""
    source = SOURCE_FILE
    facts: list[Fact] = []
    quantization = gguf.file_type_name(dict(metadata))
    if quantization is not None:
        facts.append(_described("quantization", quantization, "quantization", source))

    architecture_name = _text(metadata.get("general.architecture"))
    context = _positive_int(metadata.get(f"{architecture_name}.context_length"))
    if context is not None:
        facts.append(Fact("context_length", context, format_context(context), source))

    parameters = _positive_int(metadata.get("general.parameter_count"))
    size_label = _text(metadata.get("general.size_label"))
    if parameters is not None:
        facts.append(Fact("parameter_count", parameters, format_count(parameters), source))
    elif size_label is not None:
        facts.append(Fact("parameter_count", size_label, size_label, source))

    if architecture_name is not None:
        experts = _positive_int(metadata.get(f"{architecture_name}.expert_count"))
        kind = "moe" if experts else "dense"
        facts.append(_described("architecture", kind, "architecture", source))

    template = _text(metadata.get("tokenizer.chat_template"))
    if template is not None:
        flag = "tools" in template
        facts.append(
            Fact("tools", flag, f"{_yes_no(flag)} (judged from the chat template)", source)
        )

    license_name = _text(metadata.get("general.license"))
    if license_name:
        facts.append(Fact("license", license_name, license_name, source))

    count = _positive_int(metadata.get("general.base_model.count")) or 0
    names: list[str] = []
    for index in range(min(count, 8)):
        name = _text(metadata.get(f"general.base_model.{index}.name"))
        owner = _text(metadata.get(f"general.base_model.{index}.organization"))
        if name is not None:
            names.append(f"{name} ({owner})" if owner else name)
    if names:
        value = {"models": names, "relation": None}
        facts.append(Fact("base_model", value, ", ".join(names), source))

    if size is not None:
        facts.append(Fact("file_size", size, format_bytes(size), source))
    return facts


def _lookup_file(name: str, parsed: ParsedName) -> tuple[str, list[Fact]]:
    """Read the header of ``name`` if it is a GGUF file that exists on this machine."""
    if KIND_FILE not in parsed.kinds:
        return STATUS_NOT_APPLICABLE, []
    try:
        path = Path(name.strip()).expanduser()
        if not path.is_file():
            return STATUS_NOT_APPLICABLE, []  # just a file name, with no file behind it
        metadata = gguf.read_metadata(path)
        size = path.stat().st_size
    except (OSError, ValueError):
        return STATUS_ERROR, []
    return STATUS_OK, gguf_facts(metadata, size)


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

    based_on = _base_model(card, tags)
    if based_on is not None:
        facts.append(based_on)

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


# -------------------------------------------------------------- openrouter ----


def _openrouter_id(parsed: ParsedName) -> str | None:
    """The OpenRouter model ID this name could be, or ``None`` if it is not one."""
    if parsed.namespace is None:
        return None
    base = f"{parsed.namespace}/{parsed.model}".lower()
    if parsed.kinds == (KIND_OPENROUTER,):
        return f"{base}:{parsed.tag.lower()}" if parsed.tag else base
    if parsed.kinds == (KIND_API,):
        return base
    return None


def openrouter_facts(entry: Mapping[str, Any]) -> list[Fact]:
    """Turn one entry of OpenRouter's model list into facts."""
    source = SOURCE_OPENROUTER
    facts: list[Fact] = []
    context = _positive_int(entry.get("context_length"))
    if context is not None:
        facts.append(Fact("context_length", context, format_context(context), source))
    architecture = entry.get("architecture")
    inputs = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    if isinstance(inputs, list) and inputs:
        flag = "image" in inputs
        facts.append(Fact("vision", flag, _yes_no(flag), source))
    supported = entry.get("supported_parameters")
    if isinstance(supported, list) and supported:
        for key, parameter in (("tools", "tools"), ("thinking", "reasoning")):
            flag = parameter in supported
            facts.append(Fact(key, flag, _yes_no(flag), source))
    return facts


def _lookup_openrouter(
    parsed: ParsedName, client: httpx.Client, timeout: float, cache_dir: Path | None = None
) -> tuple[str, list[Fact], bool]:
    """Find the name in OpenRouter's model list. The name itself is never sent."""
    wanted = _openrouter_id(parsed)
    if wanted is None:
        return STATUS_NOT_APPLICABLE, [], False
    body = cache.read(cache_dir, _OPENROUTER_CACHE_KEY) if cache_dir is not None else None
    cached = body is not None
    if body is None:
        status, body = _request(client, "GET", OPENROUTER_URL, timeout=timeout)
        if status != STATUS_OK:
            return status, [], False
        if cache_dir is not None:
            cache.write(cache_dir, _OPENROUTER_CACHE_KEY, body)
    models = body.get("data")
    listed = {
        item["id"].lower(): item
        for item in (models if isinstance(models, list) else [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    # A variant such as ":free" is often not listed on its own; the plain ID then answers.
    entry = listed.get(wanted) or listed.get(wanted.split(":", 1)[0])
    if entry is None:
        return STATUS_NOT_FOUND, [], False
    return STATUS_OK, openrouter_facts(entry), cached


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

    ``offline=True`` makes no request. A GGUF file that exists on this machine
    is still read, since that needs no network. ``timeout``
    overrides the per-source defaults. ``client`` lets a caller share or mock
    the HTTP client. ``cache_dir`` turns on a one-day on-disk cache of Hugging
    Face answers; without it nothing is read from or written to disk. Lookup
    failures never raise.
    """
    parsed = parse(name)
    # A file on this machine is read whether or not the network may be used.
    file_status, from_file = _lookup_file(name, parsed)
    on_disk = {SOURCE_FILE: file_status} if file_status != STATUS_NOT_APPLICABLE else {}
    if offline:
        skipped = {SOURCE_OLLAMA: STATUS_SKIPPED, SOURCE_HUGGINGFACE: STATUS_SKIPPED}
        return ModelInfo(parsed, tuple(from_file), {**skipped, **on_disk})

    wants_ollama = _ollama_name(parsed) is not None
    wants_hub = _huggingface_repo(parsed) is not None
    wants_router = _openrouter_id(parsed) is not None
    if not (wants_ollama or wants_hub or wants_router):
        nothing = {SOURCE_OLLAMA: STATUS_NOT_APPLICABLE, SOURCE_HUGGINGFACE: STATUS_NOT_APPLICABLE}
        return ModelInfo(parsed, tuple(from_file), {**nothing, **on_disk})

    owned = client is None
    active = client or _new_client()
    try:
        ollama_status, from_ollama = _lookup_ollama(
            parsed, active, ollama_url, OLLAMA_TIMEOUT if timeout is None else timeout
        )
        hub_status, from_hub, hub_cached = _lookup_huggingface(
            parsed, active, HUGGINGFACE_TIMEOUT if timeout is None else timeout, cache_dir
        )
        router_status, from_router, router_cached = _lookup_openrouter(
            parsed, active, OPENROUTER_TIMEOUT if timeout is None else timeout, cache_dir
        )
    finally:
        if owned:
            active.close()
    lookups = {SOURCE_OLLAMA: ollama_status, SOURCE_HUGGINGFACE: hub_status, **on_disk}
    if router_status != STATUS_NOT_APPLICABLE:
        lookups[SOURCE_OPENROUTER] = router_status
    cached = tuple(
        source
        for source, used in ((SOURCE_HUGGINGFACE, hub_cached), (SOURCE_OPENROUTER, router_cached))
        if used
    )
    return ModelInfo(
        parsed, _merge([from_file, from_ollama, from_hub, from_router]), lookups, cached
    )
