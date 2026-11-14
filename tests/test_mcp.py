"""The MCP server."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from whatmodel import explain
from whatmodel.cli import app
from whatmodel.lookup import ModelInfo
from whatmodel.mcp import SUPPORTED_VERSIONS, TOOLS, Server, serve


def offline(name: str, **options: Any) -> ModelInfo:
    """Stand in for explain() so no test here can reach the network."""
    return explain(name, offline=True)


@pytest.fixture
def server() -> Server:
    return Server(explain_name=offline)


def ask(server: Server, method: str, params: Any = None, request_id: int = 1) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    answer = server.handle(message)
    assert answer is not None
    assert answer["jsonrpc"] == "2.0"
    assert answer["id"] == request_id
    return answer


def call(server: Server, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return ask(server, "tools/call", {"name": tool, "arguments": arguments})["result"]


def test_older_clients_open_with_initialize(server: Server) -> None:
    params = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}
    result = ask(server, "initialize", params)["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"]["name"] == "whatmodel"


def test_initialize_with_an_unknown_version_gets_one_we_speak(server: Server) -> None:
    result = ask(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert result["protocolVersion"] == "2025-11-25"


def test_newer_clients_can_discover(server: Server) -> None:
    result = ask(server, "server/discover", {"_meta": {}})["result"]
    assert result["supportedVersions"] == list(SUPPORTED_VERSIONS)
    assert result["resultType"] == "complete"
    assert result["capabilities"] == {"tools": {}}
    assert isinstance(result["ttlMs"], int)


def test_an_unsupported_version_in_a_request_is_refused(server: Server) -> None:
    meta = {"_meta": {"io.modelcontextprotocol/protocolVersion": "2099-01-01"}}
    error = ask(server, "tools/list", meta)["error"]
    assert error["code"] == -32022
    assert error["data"] == {"supported": list(SUPPORTED_VERSIONS), "requested": "2099-01-01"}


def test_tools_are_listed(server: Server) -> None:
    result = ask(server, "tools/list")["result"]
    assert [tool["name"] for tool in result["tools"]] == [
        "explain_model_name",
        "compare_model_names",
    ]
    for tool in TOOLS:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


def test_explain_tool(server: Server) -> None:
    result = call(server, "explain_model_name", {"name": "gemma4:e2b-it-qat"})
    assert result["resultType"] == "complete"
    assert "isError" not in result
    (block,) = result["content"]
    assert block["type"] == "text"
    assert "- Family: Gemma 4 (Google) [name]" in block["text"]
    assert result["structuredContent"]["input"] == "gemma4:e2b-it-qat"


def test_explain_tool_passes_on_notes_and_unknowns(server: Server) -> None:
    text = call(server, "explain_model_name", {"name": "qwen-max-zzz"})["content"][0]["text"]
    assert "- Unknown: zzz [name]" in text
    assert "Note: No tag in the name." in text


def test_compare_tool(server: Server) -> None:
    names = ["llama3.1:8b-instruct-q4_K_M", "llama3.1:8b-instruct-q5_K_M"]
    result = call(server, "compare_model_names", {"names": names})
    text = result["content"][0]["text"]
    assert "- Family (from the name, same):" in text
    assert "- Compression (from the name, differs):" in text
    assert [model["input"] for model in result["structuredContent"]["models"]] == names


def test_lookup_flag_reaches_explain() -> None:
    seen: list[dict[str, Any]] = []

    def spy(name: str, **options: Any) -> ModelInfo:
        seen.append(options)
        return explain(name, offline=True)

    server = Server(explain_name=spy)
    call(server, "explain_model_name", {"name": "qwen3.5:4b"})
    call(server, "explain_model_name", {"name": "qwen3.5:4b", "lookup": False})
    assert [options["offline"] for options in seen] == [False, True]


def test_a_name_that_cannot_be_read_is_a_tool_error_the_model_can_see(server: Server) -> None:
    result = call(server, "explain_model_name", {"name": "/"})
    assert result["isError"] is True
    assert "Not a model name" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("explain_model_name", {}),
        ("explain_model_name", {"name": "  "}),
        ("explain_model_name", {"name": "x", "lookup": "yes"}),
        ("compare_model_names", {"names": ["only-one"]}),
        ("compare_model_names", {"names": "a,b"}),
        ("no_such_tool", {}),
    ],
)
def test_bad_arguments_are_protocol_errors(
    server: Server, tool: str, arguments: dict[str, Any]
) -> None:
    answer = ask(server, "tools/call", {"name": tool, "arguments": arguments})
    assert answer["error"]["code"] == -32602


def test_unknown_method_and_ping(server: Server) -> None:
    assert ask(server, "resources/list")["error"]["code"] == -32601
    assert ask(server, "ping")["result"] == {"resultType": "complete"}


def test_notifications_get_no_answer(server: Server) -> None:
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None


def test_malformed_messages_are_errors_not_crashes(server: Server) -> None:
    for message in ("text", [], {"id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": 1}):
        answer = server.handle(message)
        assert answer is not None
        assert answer["error"]["code"] == -32600


def test_a_failure_inside_a_tool_does_not_end_the_session() -> None:
    def broken(name: str, **options: Any) -> ModelInfo:
        raise RuntimeError("boom")

    answer = ask(
        Server(explain_name=broken),
        "tools/call",
        {
            "name": "explain_model_name",
            "arguments": {"name": "qwen3.5:4b"},
        },
    )
    assert answer["error"]["code"] == -32603
    assert "boom" not in json.dumps(answer)  # details stay out of the reply


def test_serve_reads_and_writes_one_message_per_line(server: Server) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        "",
        "not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    writer = io.StringIO()
    serve(server, reader=io.StringIO("\n".join(lines) + "\n"), writer=writer)
    answers = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert [answer.get("id") for answer in answers] == [1, None, 2]
    assert answers[1]["error"]["code"] == -32700


def test_mcp_command_is_registered() -> None:
    from typer.testing import CliRunner

    lines = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    result = CliRunner().invoke(app, ["mcp", "--no-cache"], input=lines)
    assert result.exit_code == 0
    assert json.loads(result.output)["result"] == {"resultType": "complete"}
