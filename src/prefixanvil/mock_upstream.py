"""Deterministic mock upstream (OpenAI-compatible) for tests and `prefixanvil equiv`.

It behaves like a temperature-0 "model" whose output is a pure function of the
*semantic* content of the request under an explicit, declared equivalence model
(``prefixanvil.mock-upstream/v1``). It is NOT a model: it cannot tell whether a
real LLM would answer differently. Real-model equivalence is a Phase 3
measurement.

Declared equivalence model (what the mock treats as NOT meaningful):

* JSON object key order and JSON whitespace/separators, wherever a value is
  JSON: tool definitions, ``tool_calls[].function.arguments`` (parsed one level:
  a string *inside* the arguments stays a verbatim string), and tool-role
  content that parses as a JSON object/array.
* Order of the ``tools`` array (tools are a multiset of canonical definitions;
  duplicates still count).
* In system/developer text, *outside* fenced code blocks: trailing whitespace on
  a line, runs of spaces/tabs after the first non-blank character (except on
  lines indented by >= 4 spaces / a tab, i.e. Markdown code), 3+ newlines vs 2,
  and trailing newlines.
* Transport fields: ``stream``, ``stream_options``, ``session_id``, ``metadata``.

Everything else is meaningful, including: leading indentation, anything inside
``` fences, spacing inside indented code lines, line order, user/assistant text, message order, tool_call order,
non-JSON tool content, and sampling parameters.

Output: one 4-word group per semantic element, in request order (system
messages, then the tools multiset, then the remaining messages, then
parameters). The first differing word therefore points at the first element
whose semantics changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "prefixanvil.mock-upstream/v1"
_WORDS = ("amber birch cedar delta ember fjord grove harbor island juniper kelp lumen "
          "meadow nectar orchid prism quartz raven sierra tundra umber violet willow "
          "xenon yarrow zephyr anchor beacon canyon dune estuary falcon").split()
_TRANSPORT = {"stream", "stream_options", "session_id", "metadata", "messages", "tools"}
_FENCE = re.compile(r"^\s*(```|~~~)")


def _canon_json(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _norm_system(text: str) -> str:
    out, in_fence, prev_blank = [], False, False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line.rstrip())
            prev_blank = False
            continue
        if in_fence:
            out.append(line)
            continue
        line = line.rstrip()
        if not line:
            if prev_blank:          # 3+ newlines == 2 newlines (outside fences)
                continue
            prev_blank = True
            out.append("")
            continue
        prev_blank = False
        if re.match(r"^(?: {4}|\t)", line):   # indented (Markdown) code: spacing is meaningful
            out.append(line)
            continue
        m = re.match(r"^(\s*)(.*)$", line)
        out.append(m.group(1) + re.sub(r"[ \t]{2,}", " ", m.group(2)))
    return "\n".join(out).rstrip("\n")


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) and p.get("type") == "text"
                       else f"<{p.get('type') if isinstance(p, dict) else '?'}>" for p in content)
    return "" if content is None else str(content)


def _maybe_json(s: str) -> str:
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError, TypeError):
        return "S:" + s
    return "J:" + _canon_json(v) if isinstance(v, (dict, list)) else "S:" + s


def semantic_elements(body: dict) -> list[tuple[str, str]]:
    msgs = body.get("messages") or []
    sys_part, rest = [], []
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role in ("system", "developer"):
            sys_part.append((f"msg{i}:{role}", _norm_system(_text(m.get("content")))))
            continue
        c = _text(m.get("content"))
        if role == "tool":
            c = _maybe_json(c)
        calls = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            calls.append([fn.get("name"), _maybe_json(args) if isinstance(args, str) else "J:" + _canon_json(args),
                          tc.get("id")])
        rest.append((f"msg{i}:{role}", _canon_json([c, calls, m.get("tool_call_id"), m.get("name")])))
    tools = sorted(_canon_json(t) for t in (body.get("tools") or []))
    params = {k: v for k, v in body.items() if k not in _TRANSPORT}
    return sys_part + [("tools", _canon_json(tools))] + rest + [("params", _canon_json(params))]


def mock_completion_text(body: dict) -> str:
    words = []
    for tag, val in semantic_elements(body):
        h = hashlib.sha256((tag + "\x00" + val).encode("utf-8")).digest()
        words += [_WORDS[b % len(_WORDS)] for b in h[:4]]
    return " ".join(words)


def completion_response(body: dict) -> dict:
    text = mock_completion_text(body)
    rid = hashlib.sha256(text.encode()).hexdigest()[:24]
    n = len(text.split())
    return {"id": f"chatcmpl-mock-{rid}", "object": "chat.completion", "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n}}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "_Server"

    def log_message(self, *a):  # quiet
        pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self) -> None:
        raw = self._body()
        self.server.owner.received.append({"method": self.command, "path": self.path,
                                           "headers": dict(self.headers), "body": raw})
        forced = self.headers.get("X-Mock-Status")
        if forced:
            return self._json(int(forced), {"error": {"message": "forced by X-Mock-Status",
                                                      "type": "mock_error"}})
        if self.command == "POST" and self.path.split("?")[0] == "/v1/chat/completions":
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                return self._json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request"}})
            if body.get("stream"):
                return self._stream(body)
            return self._json(200, completion_response(body))
        if self.command == "GET" and self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        return self._json(200, {"echo": {"method": self.command, "path": self.path,
                                         "body_sha256": hashlib.sha256(raw).hexdigest()}})

    def _stream(self, body: dict) -> None:
        text = mock_completion_text(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        words = text.split(" ")
        for i, w in enumerate(words):
            chunk = {"id": "chatcmpl-mock-stream", "object": "chat.completion.chunk", "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": {"content": (w if i == 0 else " " + w)},
                                  "finish_reason": None}]}
            self._chunk(f"data: {json.dumps(chunk)}\n\n".encode())
        self._chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _handle


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    owner: "MockUpstream"


class MockUpstream:
    """In-process mock server on 127.0.0.1 (random port by default)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._srv = _Server((host, port), _Handler)
        self._srv.owner = self
        self.received: list[dict] = []
        self._t: threading.Thread | None = None

    @property
    def url(self) -> str:
        h, p = self._srv.server_address[:2]
        return f"http://{h}:{p}"

    def start(self) -> "MockUpstream":
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def __enter__(self) -> "MockUpstream":
        return self.start()

    def __exit__(self, *a) -> None:
        self.stop()
