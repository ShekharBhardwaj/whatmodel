"""A small Model Context Protocol server, so AI assistants can ask whatmodel.

It speaks MCP over standard input and output: one JSON-RPC message per line.
Nothing is installed for it; it uses only the standard library.

Two generations of the protocol are supported, because clients are split
between them:

- 2026-07-28, where a client may call ``server/discover`` and every result
  carries ``resultType``.
- 2025-11-25 and earlier, where a client opens with ``initialize``.

Message shapes follow schema/2026-07-28/schema.ts in
modelcontextprotocol/modelcontextprotocol, checked in November 2026.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

from whatmodel.lookup import DEFAULT_OLLAMA_URL, ModelInfo, explain
from whatmodel.render import compare_rows, display_name, info_notes, lookup_rows, name_rows

LATEST_VERSION = "2026-07-28"
#: Versions that open with ``initialize``, newest first.
HANDSHAKE_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SUPPORTED_VERSIONS = (LATEST_VERSION, *HANDSHAKE_VERSIONS)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNSUPPORTED_PROTOCOL_VERSION = -32022

_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"

#: How long a client may reuse the tool list, in milliseconds. It never changes while running.
_TTL_MS = 60 * 60 * 1000

INSTRUCTIONS = (
    "Explains what the parts of an LLM model name mean, such as 'qwen3.5:4b' or "
    "'bartowski/Qwen_Qwen3-30B-A3B-GGUF'. Rows marked 'name' are read from the name itself, "
    "rows marked 'name (implied)' are inferred from another part of the name, and the rest "
    "are looked up. A token reported as unknown has no known meaning; do not guess one."
)

_LOOKUP = {
    "type": "boolean",
    "description": (
        "Also look up facts the name does not show, from a local Ollama, Hugging Face and "
        "OpenRouter. Defaults to true. Set false to read the name only, with no network use."
    ),
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "explain_model_name",
        "title": "Explain a model name",
        "description": (
            "Explain an LLM model name in plain English: family, version, size, tuning, "
            "compression, format and more. Accepts Ollama names, Hugging Face repos, GGUF "
            "file names, hosted API model IDs and OpenRouter IDs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The model name to explain."},
                "lookup": _LOOKUP,
            },
            "required": ["name"],
        },
    },
    {
        "name": "compare_model_names",
        "title": "Compare model names",
        "description": (
            "Put two or more model names side by side, field by field, and say which fields differ."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "The model names to compare.",
                },
                "lookup": _LOOKUP,
            },
            "required": ["names"],
        },
    },
]


class _RequestError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


Explain = Callable[..., ModelInfo]


def _source(row_source: str) -> str:
    from whatmodel.render import SOURCE_LABELS

    return SOURCE_LABELS.get(row_source, row_source)


def describe(info: ModelInfo) -> str:
    """One model as plain text, a line per field, for an assistant to read."""
    lines = [display_name(info.parsed)]
    for row in [*name_rows(info.parsed), *lookup_rows(info)]:
        value = "; ".join(label for label, _ in row.entries)
        lines.append(f"- {row.heading}: {value} [{_source(row.source)}]")
    lines.extend(f"Note: {note}" for note in info_notes(info))
    return "\n".join(lines)


def describe_comparison(infos: list[ModelInfo]) -> str:
    names = [display_name(info.parsed) for info in infos]
    lines = ["Comparing: " + " | ".join(names)]
    for heading, looked_up, values in compare_rows(infos):
        cells = " | ".join(value if value is not None else "-" for value in values)
        verdict = "differs" if len(set(values)) > 1 else "same"
        origin = "looked up" if looked_up else "from the name"
        lines.append(f"- {heading} ({origin}, {verdict}): {cells}")
    return "\n".join(lines)


class Server:
    """Handles one MCP message at a time. Holds no state between messages."""

    def __init__(
        self,
        *,
        explain_name: Explain = explain,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        cache_dir: Path | None = None,
    ) -> None:
        self._explain = explain_name
        self._ollama_url = ollama_url
        self._cache_dir = cache_dir

    # ------------------------------------------------------------ protocol ----

    def handle(self, message: object) -> dict[str, Any] | None:
        """Answer one decoded message. Returns ``None`` when no answer is due."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message.")
        method, request_id = message.get("method"), message.get("id")
        if not isinstance(method, str):
            # A response to something we never asked; there is nothing to do with it.
            return (
                None
                if "result" in message or "error" in message
                else _error(request_id, INVALID_REQUEST, "The message has no method.")
            )
        if "id" not in message:
            return None  # a notification, such as notifications/initialized
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        try:
            self._check_version(params)
            result = self._dispatch(method, params)
        except _RequestError as exc:
            return _error(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # never let one bad request end the session
            return _error(request_id, INTERNAL_ERROR, f"Internal error: {type(exc).__name__}.")
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resultType": "complete", **result}}

    def _check_version(self, params: Mapping[str, Any]) -> None:
        meta = params.get("_meta")
        requested = meta.get(_VERSION_KEY) if isinstance(meta, dict) else None
        if requested is not None and requested not in SUPPORTED_VERSIONS:
            raise _RequestError(
                UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version.",
                {"supported": list(SUPPORTED_VERSIONS), "requested": requested},
            )

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "server/discover":
            return {
                "supportedVersions": list(SUPPORTED_VERSIONS),
                "capabilities": {"tools": {}},
                "instructions": INSTRUCTIONS,
                "ttlMs": _TTL_MS,
                "cacheScope": "public",
                "_meta": {_SERVER_INFO_KEY: _server_info()},
            }
        if method == "initialize":
            wanted = params.get("protocolVersion")
            version = wanted if wanted in HANDSHAKE_VERSIONS else HANDSHAKE_VERSIONS[0]
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": _server_info(),
                "instructions": INSTRUCTIONS,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS, "ttlMs": _TTL_MS, "cacheScope": "public"}
        if method == "tools/call":
            return self._call(params)
        raise _RequestError(METHOD_NOT_FOUND, f"Method not found: {method}")

    # --------------------------------------------------------------- tools ----

    def _call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name, arguments = params.get("name"), params.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        lookup = arguments.get("lookup", True)
        if not isinstance(lookup, bool):
            raise _RequestError(INVALID_PARAMS, "'lookup' must be true or false.")
        if name == "explain_model_name":
            model = arguments.get("name")
            if not isinstance(model, str) or not model.strip():
                raise _RequestError(INVALID_PARAMS, "'name' must be a model name.")
            return self._answer([model], lookup, single=True)
        if name == "compare_model_names":
            models = arguments.get("names")
            if (
                not isinstance(models, list)
                or len(models) < 2
                or not all(isinstance(model, str) and model.strip() for model in models)
            ):
                raise _RequestError(INVALID_PARAMS, "'names' must list two or more model names.")
            return self._answer(models, lookup, single=False)
        raise _RequestError(INVALID_PARAMS, f"Unknown tool: {name}")

    def _answer(self, names: list[str], lookup: bool, *, single: bool) -> dict[str, Any]:
        try:
            infos = [
                self._explain(
                    name,
                    offline=not lookup,
                    ollama_url=self._ollama_url,
                    cache_dir=self._cache_dir if lookup else None,
                )
                for name in names
            ]
        except ValueError as exc:
            # A name that cannot be read is the caller's to fix, so the model must see it.
            return {
                "content": [{"type": "text", "text": f"Not a model name: {exc}."}],
                "isError": True,
            }
        if single:
            text, data = describe(infos[0]), infos[0].to_dict()
        else:
            text = describe_comparison(infos)
            data = {"models": [info.to_dict() for info in infos]}
        return {"content": [{"type": "text", "text": text}], "structuredContent": data}


def _server_info() -> dict[str, str]:
    from whatmodel import __version__

    return {"name": "whatmodel", "title": "whatmodel", "version": __version__}


def _error(request_id: object, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def serve(
    server: Server | None = None, *, reader: TextIO | None = None, writer: TextIO | None = None
) -> None:
    """Read messages from ``reader`` until it closes, writing one answer per line."""
    server = server or Server()
    reader = reader or sys.stdin
    writer = writer or sys.stdout
    for line in reader:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            answer: dict[str, Any] | None = _error(None, PARSE_ERROR, "Invalid JSON.")
        else:
            answer = server.handle(message)
        if answer is not None:
            # One compact line per message: embedded newlines would break the framing.
            writer.write(json.dumps(answer, separators=(",", ":"), ensure_ascii=False) + "\n")
            writer.flush()


__all__ = ["SUPPORTED_VERSIONS", "TOOLS", "Server", "describe", "describe_comparison", "serve"]
