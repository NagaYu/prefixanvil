"""`prefixanvil proxy`: OpenAI-compatible HTTP proxy (stdlib only).

Routes: ``POST /v1/chat/completions`` (the only transformed route),
``GET /health``, and transparent passthrough for every other path/method.

Configuration (environment):

* ``PREFIXANVIL_UPSTREAM_BASE_URL`` (required) e.g. ``http://127.0.0.1:8000``
* ``PREFIXANVIL_PORT`` (default 8080), ``PREFIXANVIL_HOST`` (default 127.0.0.1)
* ``PREFIXANVIL_MODE`` ``L0`` (default) | ``L1`` | ``L2``
* ``PREFIXANVIL_L2_ENABLE=1`` additionally required for L2; L2 also requires a
  passing equivalence report for the current build at ``PREFIXANVIL_EQUIV_REPORT``
  (default ``./equivalence-report.json``) or the proxy refuses to start.
* ``PREFIXANVIL_LOG`` path of the per-request JSONL transform log (unset = no log)
* ``PREFIXANVIL_L1_TOOL_ORDER`` ``session`` (default) | ``stateless`` — ``stateless``
  passes no session context to the core, i.e. the Phase-1 name sort (used by
  the cross-session benchmark to price the policy choice)
* ``PREFIXANVIL_TOKENIZER`` tokenizer dir for L0 on-the-fly classification
  (default: bundled fixture tokenizer), ``PREFIXANVIL_MAX_SESSIONS`` (default 10000)

Session identity: header ``X-PrefixAnvil-Session-Id``, else body ``session_id``,
else body ``metadata.session_id``. Without one, L1 uses the stateless sort.

The proxy is a thin caller of :mod:`prefixanvil.core`; all per-session state
(SessionContext JSON, previous request for L0 pairing) lives here, in a
bounded LRU map.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .core import OBSERVE_REQUEST_SCHEMA, TRANSFORM_REQUEST_SCHEMA, MODES, Observer, transform

CHAT_PATH = "/v1/chat/completions"
LOG_SCHEMA = "prefixanvil.proxy-log/v1"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class ConfigError(Exception):
    pass


@dataclass
class ProxyConfig:
    upstream: str
    port: int = 8080
    host: str = "127.0.0.1"
    mode: str = "L0"
    l2_enable: bool = False
    log_path: str | None = None
    equiv_report: str = "equivalence-report.json"
    tokenizer: str | None = None
    l1_tool_order: str = "session"         # "session" (Phase 2 default) | "stateless" (Phase 1 name sort)
    max_sessions: int = 10000
    timeout_s: float = 600.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ProxyConfig":
        env = os.environ if env is None else env
        up = env.get("PREFIXANVIL_UPSTREAM_BASE_URL")
        if not up:
            raise ConfigError("PREFIXANVIL_UPSTREAM_BASE_URL is required")
        u = urlsplit(up)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ConfigError(f"PREFIXANVIL_UPSTREAM_BASE_URL must be an http(s) URL, got {up!r}")
        try:
            port = int(env.get("PREFIXANVIL_PORT", "8080"))
            max_sessions = int(env.get("PREFIXANVIL_MAX_SESSIONS", "10000"))
        except ValueError as e:
            raise ConfigError(f"invalid integer in environment: {e}") from None
        mode = env.get("PREFIXANVIL_MODE", "L0").upper()
        if mode not in MODES:
            raise ConfigError(f"PREFIXANVIL_MODE must be one of {MODES}, got {mode!r}")
        return cls(upstream=up.rstrip("/"), port=port, host=env.get("PREFIXANVIL_HOST", "127.0.0.1"),
                   mode=mode, l2_enable=env.get("PREFIXANVIL_L2_ENABLE") == "1",
                   log_path=env.get("PREFIXANVIL_LOG") or None,
                   equiv_report=env.get("PREFIXANVIL_EQUIV_REPORT", "equivalence-report.json"),
                   tokenizer=env.get("PREFIXANVIL_TOKENIZER") or None, max_sessions=max_sessions,
                   l1_tool_order=_tool_order(env.get("PREFIXANVIL_L1_TOOL_ORDER", "session")))


def _tool_order(v: str) -> str:
    v = v.strip().lower()
    if v not in ("session", "stateless"):
        raise ConfigError(f"PREFIXANVIL_L1_TOOL_ORDER must be 'session' or 'stateless', got {v!r}")
    return v


def preflight(cfg: ProxyConfig) -> None:
    """Refuse to start in L2 unless explicitly enabled AND the equivalence gate passed."""
    if cfg.mode != "L2":
        return
    if not cfg.l2_enable:
        raise ConfigError("PREFIXANVIL_MODE=L2 requires PREFIXANVIL_L2_ENABLE=1")
    from .equiv import check_report
    ok, msg = check_report(cfg.equiv_report, "L2")
    if not ok:
        raise ConfigError(f"L2 refused: {msg}")


class _Sessions:
    def __init__(self, max_items: int):
        self.max = max_items
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._last_start: dict[str, dict] = {}     # model -> first request of the latest session
        self.lock = threading.Lock()

    def get(self, sid: str) -> dict | None:
        with self.lock:
            v = self._d.get(sid)
            if v is not None:
                self._d.move_to_end(sid)
            return v

    def put(self, sid: str, v: dict) -> None:
        with self.lock:
            self._d[sid] = v
            self._d.move_to_end(sid)
            while len(self._d) > self.max:
                self._d.popitem(last=False)


def _hash_id(s: str | None) -> str | None:
    return None if s is None else hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _session_id(headers, body: dict | None) -> str | None:
    sid = headers.get("X-PrefixAnvil-Session-Id")
    if not sid and isinstance(body, dict):
        sid = body.get("session_id")
        if not sid and isinstance(body.get("metadata"), dict):
            sid = body["metadata"].get("session_id")
    return str(sid) if sid not in (None, "") else None


class AnvilProxy:
    def __init__(self, cfg: ProxyConfig, observer: Observer | None = None):
        self.cfg = cfg
        self.sessions = _Sessions(cfg.max_sessions)
        self._log_lock = threading.Lock()
        self._observer = observer
        self._observer_lock = threading.Lock()
        u = urlsplit(cfg.upstream)
        self._up_scheme, self._up_host, self._up_port = u.scheme, u.hostname, u.port
        self._up_prefix = u.path.rstrip("/")
        self._srv = _Server((cfg.host, cfg.port), _Handler)
        self._srv.proxy = self
        self._t: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------
    @property
    def url(self) -> str:
        h, p = self._srv.server_address[:2]
        return f"http://{h}:{p}"

    def start(self) -> "AnvilProxy":
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def serve_forever(self) -> None:
        self._srv.serve_forever()

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    # -- helpers ----------------------------------------------------------
    def observer(self) -> Observer:
        with self._observer_lock:
            if self._observer is None:
                from .render import Renderer, load_tokenizer
                self._observer = Observer(Renderer(load_tokenizer(self.cfg.tokenizer)))
            return self._observer

    def upstream_conn(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._up_scheme == "https" else http.client.HTTPConnection
        return cls(self._up_host, self._up_port, timeout=self.cfg.timeout_s)

    def write_log(self, entry: dict) -> None:
        if not self.cfg.log_path:
            return
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._log_lock, open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    # -- the transformed route -------------------------------------------
    def prepare_chat(self, raw: bytes, headers) -> tuple[bytes, dict]:
        """Return (bytes to forward, log fields). Never raises (fail-open)."""
        log: dict[str, Any] = {"mode": self.cfg.mode, "applied": [], "skipped": [], "notes": []}
        try:
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                log["skipped"].append({"rule": "*", "path": "request", "reason": "invalid_json"})
                log["before_sha256"] = log["after_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
                return raw, log
            sid = _session_id(headers, body)
            log["session"] = _hash_id(sid)
            log["stream"] = bool(isinstance(body, dict) and body.get("stream"))
            state = self.sessions.get(sid) if sid else None
            ctx = (state or {}).get("ctx") if sid else None
            if sid and ctx is None:
                ctx = {"schema": "prefixanvil.session-context/v1", "prev_tool_names": None,
                       "prev_tool_order": None}
            res = transform({"schema": TRANSFORM_REQUEST_SCHEMA, "mode": self.cfg.mode,
                             "request": body,
                             "session_context": ctx if sid and self.cfg.l1_tool_order == "session" else None})
            log.update(before_sha256=res["before_sha256"], after_sha256=res["after_sha256"],
                       changed=res["changed"], applied=res["applied"], skipped=res["skipped"],
                       notes=res["notes"])
            if self.cfg.mode == "L0":
                log["would_apply"] = res.get("would_apply", [])
                log["observe"] = self._observe(sid, body, state)
                out = raw                                   # byte-identical forwarding
            else:
                out = json.dumps(res["request"], ensure_ascii=False).encode("utf-8") if res["changed"] else raw
            if sid and not any(s["rule"] == "*" for s in res["skipped"]):
                self.sessions.put(sid, {"ctx": res["session_context"],
                                        "prev": body if self.cfg.mode == "L0" else None})
            return out, log
        except Exception as e:
            log["error"] = f"internal_error:{type(e).__name__}"
            log["skipped"].append({"rule": "*", "path": "request", "reason": log["error"]})
            log["before_sha256"] = log["after_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
            return raw, log

    def _observe(self, sid: str | None, body: dict, state: dict | None) -> dict:
        if not isinstance(body, dict):
            return {"label": None, "detail": "not_an_object"}
        scope, ref = "within_session", (state or {}).get("prev")
        model = body.get("model")
        if sid is None:
            return {"label": None, "detail": "no_session_id"}
        if ref is None:
            with self.sessions.lock:
                ref = self.sessions._last_start.get(model)
                self.sessions._last_start[model] = body
            scope = "cross_session"
        if ref is None:
            return {"label": None, "detail": "no_reference"}
        r = self.observer().observe({"schema": OBSERVE_REQUEST_SCHEMA, "request": body,
                                     "reference": ref, "scope": scope})
        r.pop("schema", None)
        return r


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    proxy: AnvilProxy


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, *a):  # requests are logged (hash-only) by AnvilProxy.write_log
        pass

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send_json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self) -> None:
        px = self.server.proxy
        path = self.path.split("?", 1)[0]
        if self.command == "GET" and path == "/health":
            return self._send_json(200, {"status": "ok", "mode": px.cfg.mode})
        t0 = time.perf_counter()
        raw = self._read_body()
        entry: dict[str, Any] = {"schema": LOG_SCHEMA, "ts": round(time.time(), 3),
                                 "method": self.command, "path": path}
        if self.command == "POST" and path == CHAT_PATH:
            out, fields = px.prepare_chat(raw, self.headers)
            entry.update(fields)
        else:
            out = raw
            entry.update(mode="passthrough")
        def on_upstream(status: int, up_ms: float | None) -> None:
            # logged once the upstream response is ready, before relaying it, so the
            # entry exists by the time the client sees the end of the response
            entry.update(upstream_status=status, upstream_ms=up_ms,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2))
            px.write_log(entry)

        self._forward(out, on_upstream)

    def _forward(self, body: bytes, on_upstream) -> None:
        px = self.server.proxy
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Content-Length"] = str(len(body))
        t0 = time.perf_counter()
        conn = px.upstream_conn()
        try:
            try:
                conn.request(self.command, px._up_prefix + self.path, body if body or self.command != "GET" else None,
                             headers)
                resp = conn.getresponse()
            except OSError as e:
                on_upstream(502, None)
                self._send_json(502, {"error": {"message": "upstream unavailable",
                                                "type": "prefixanvil_upstream_error",
                                                "code": type(e).__name__}})
                return
            up_ms = round((time.perf_counter() - t0) * 1000, 2)
            length = resp.getheader("Content-Length")
            data = resp.read() if length is not None else None
            on_upstream(resp.status, up_ms)
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            if data is not None:                # fixed-size body: relay as is
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:                               # streaming (SSE) / chunked: relay chunk by chunk
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _dispatch


def main_from_env(env: dict[str, str] | None = None) -> int:
    try:
        cfg = ProxyConfig.from_env(env)
        preflight(cfg)
        px = AnvilProxy(cfg)
    except ConfigError as e:
        print(f"prefixanvil proxy: {e}", file=sys.stderr)
        return 2
    print(f"prefixanvil proxy: mode={cfg.mode} listening on {px.url} -> {cfg.upstream}", file=sys.stderr)
    try:
        px.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
