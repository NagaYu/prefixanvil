"""prefixanvil proxy against the in-process deterministic mock upstream (localhost only)."""

import http.client
import json
from urllib.parse import urlsplit

import pytest

from prefixanvil.mock_upstream import MockUpstream
from prefixanvil.proxy import ConfigError, AnvilProxy, ProxyConfig, main_from_env, preflight

SECRET = "TOP-SECRET-PROMPT-CONTENT-7f3a"


@pytest.fixture()
def mock():
    with MockUpstream() as m:
        yield m


@pytest.fixture()
def make_proxy(mock, tmp_path):
    started = []

    def _make(mode="L0", **kw):
        cfg = ProxyConfig(upstream=mock.url, port=0, mode=mode, log_path=str(tmp_path / f"log-{mode}.jsonl"), **kw)
        px = AnvilProxy(cfg).start()
        started.append(px)
        return px

    yield _make
    for px in started:
        px.stop()


def call(url, method="POST", path="/v1/chat/completions", body=b"", headers=None):
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    try:
        c.request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        c.close()


def T(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def chat(tools=None, system="You are a helpful agent. " + SECRET, extra=None, **kw):
    b = {"model": "m", "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": "hello " + SECRET}] + (extra or [])}
    if tools is not None:
        b["tools"] = [T(n) for n in tools]
    b.update(kw)
    return b


def names(raw: bytes):
    return [t["function"]["name"] for t in json.loads(raw).get("tools", [])]


def read_log(px):
    with open(px.cfg.log_path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def test_health(make_proxy):
    px = make_proxy("L1")
    st, _, body = call(px.url, "GET", "/health")
    assert st == 200 and json.loads(body) == {"status": "ok", "mode": "L1"}


def test_l0_forwards_byte_identical_and_observes(make_proxy, mock):
    px = make_proxy("L0")
    raw1 = b'{"model":"m",   "messages":[{"role":"system","content":"Sys  x\\n\\n\\n"},{"role":"user","content":"hi"}],"tools":[{"function":{"name":"b"},"type":"function"},{"type":"function","function":{"name":"a"}}]}'
    st, _, _ = call(px.url, body=raw1, headers={"X-PrefixAnvil-Session-Id": "s1"})
    assert st == 200 and mock.received[-1]["body"] == raw1
    b2 = json.loads(raw1)
    b2["messages"][0]["content"] = "Sys  x\n"                 # whitespace drift -> F6 within session
    raw2 = json.dumps(b2).encode()
    call(px.url, body=raw2, headers={"X-PrefixAnvil-Session-Id": "s1"})
    assert mock.received[-1]["body"] == raw2
    log = read_log(px)
    assert all(e["mode"] == "L0" and e["changed"] is False for e in log)
    assert {w["rule"] for w in log[0]["would_apply"]} >= {"L1.system.whitespace", "L1.tools.sort_keys"}
    assert log[1]["observe"]["label"] == "F6" and log[1]["observe"]["scope"] == "within_session"


def test_l1_canonicalizes_with_session_order(make_proxy, mock):
    px = make_proxy("L1")
    h = {"X-PrefixAnvil-Session-Id": "sess-A"}
    call(px.url, body=json.dumps(chat(["b", "a", "c"])).encode(), headers=h)
    assert names(mock.received[-1]["body"]) == ["b", "a", "c"]          # first request: client order
    call(px.url, body=json.dumps(chat(["c", "a", "b"])).encode(), headers=h)
    assert names(mock.received[-1]["body"]) == ["b", "a", "c"]          # unchanged set: same order as before
    call(px.url, body=json.dumps(chat(["b", "a"])).encode(), headers=h)
    assert names(mock.received[-1]["body"]) == ["b", "a"]               # set changed: order untouched
    log = read_log(px)
    assert log[1]["changed"] is True
    assert any(a["rule"] == "L1.tools.session_stable_order" for a in log[1]["applied"])
    assert any(s["rule"] == "L1.tools.sort_by_name" and s["reason"] == "tool_set_changed"
               for s in log[2]["skipped"])
    # another session is independent; no session id -> stateless sort (noted)
    call(px.url, body=json.dumps(chat(["z", "y"], session_id="sess-B")).encode())
    assert names(mock.received[-1]["body"]) == ["z", "y"]
    call(px.url, body=json.dumps(chat(["z", "y"])).encode())
    assert names(mock.received[-1]["body"]) == ["y", "z"]
    assert "tool_sort:stateless_no_session_context" in read_log(px)[-1]["notes"]


def test_l1_unchanged_request_forwarded_as_is(make_proxy, mock):
    px = make_proxy("L1")
    raw = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}]}, separators=(",", ":")).encode()
    call(px.url, body=raw)
    assert mock.received[-1]["body"] == raw


def test_streaming_passthrough(make_proxy, mock):
    px = make_proxy("L1")
    body = chat(["b", "a"], stream=True)
    st, hdrs, data = call(px.url, body=json.dumps(body).encode(), headers={"X-PrefixAnvil-Session-Id": "st"})
    direct = call(mock.url, body=mock.received[-1]["body"])[2]
    assert st == 200 and hdrs.get("Content-Type") == "text/event-stream"
    assert data == direct and data.endswith(b"data: [DONE]\n\n") and data.count(b"data: ") > 3
    assert read_log(px)[-1]["stream"] is True


def test_fail_open_on_malformed_request(make_proxy, mock):
    px = make_proxy("L1")
    bad = b'{"model": "m", "messages": [ ' + SECRET.encode()
    st, _, _ = call(px.url, body=bad)
    assert mock.received[-1]["body"] == bad and st == 400            # upstream's answer passed through
    e = read_log(px)[-1]
    assert e["skipped"][0]["reason"] == "invalid_json" and e["upstream_status"] == 400
    wrong_shape = json.dumps({"model": "m", "messages": "nope"}).encode()
    call(px.url, body=wrong_shape)
    assert mock.received[-1]["body"] == wrong_shape
    assert read_log(px)[-1]["skipped"][0]["reason"] == "invalid_request:missing_messages"


def test_fail_open_on_internal_error(make_proxy, mock, monkeypatch):
    import prefixanvil.proxy as proxy_mod
    px = make_proxy("L1")
    monkeypatch.setattr(proxy_mod, "transform", lambda *a, **k: 1 / 0)
    raw = json.dumps(chat(["b", "a"])).encode()
    st, _, _ = call(px.url, body=raw)
    assert st == 200 and mock.received[-1]["body"] == raw
    assert read_log(px)[-1]["error"] == "internal_error:ZeroDivisionError"


def test_upstream_5xx_and_passthrough_routes(make_proxy, mock):
    px = make_proxy("L1")
    st, _, body = call(px.url, body=json.dumps(chat()).encode(), headers={"X-Mock-Status": "503"})
    assert st == 503 and json.loads(body)["error"]["type"] == "mock_error"
    st, _, body = call(px.url, "GET", "/v1/models")
    assert st == 200 and json.loads(body)["data"][0]["object"] == "model"
    payload = b'{"input": "embed me"}'
    st, _, body = call(px.url, "POST", "/v1/embeddings?x=1", payload)
    echo = json.loads(body)["echo"]
    assert echo["path"] == "/v1/embeddings?x=1" and mock.received[-1]["body"] == payload
    assert read_log(px)[-1]["mode"] == "passthrough"


def test_upstream_down_returns_502(tmp_path):
    px = AnvilProxy(ProxyConfig(upstream="http://127.0.0.1:9", port=0, mode="L1",
                                log_path=str(tmp_path / "l.jsonl"))).start()
    try:
        st, _, body = call(px.url, body=json.dumps(chat()).encode())
        assert st == 502 and json.loads(body)["error"]["type"] == "prefixanvil_upstream_error"
    finally:
        px.stop()


def test_log_contains_hashes_not_content(make_proxy, mock):
    for mode in ("L0", "L1"):
        px = make_proxy(mode)
        for k in range(3):
            call(px.url, body=json.dumps(chat(["b", "a"], system=f"Sys {SECRET}  \n\n\n{k}")).encode(),
                 headers={"X-PrefixAnvil-Session-Id": "secret-session-name"})
        text = open(px.cfg.log_path, encoding="utf-8").read()
        assert SECRET not in text and "hello" not in text and "secret-session-name" not in text
        e = json.loads(text.splitlines()[-1])
        for key in ("mode", "before_sha256", "after_sha256", "upstream_status", "latency_ms", "skipped"):
            assert key in e
        assert len(e["before_sha256"]) == 16


def test_config_from_env_and_l2_preflight(tmp_path):
    with pytest.raises(ConfigError):
        ProxyConfig.from_env({})
    with pytest.raises(ConfigError):
        ProxyConfig.from_env({"PREFIXANVIL_UPSTREAM_BASE_URL": "http://x", "PREFIXANVIL_MODE": "L9"})
    cfg = ProxyConfig.from_env({"PREFIXANVIL_UPSTREAM_BASE_URL": "http://127.0.0.1:8000/", "PREFIXANVIL_MODE": "l1"})
    assert (cfg.port, cfg.mode, cfg.upstream) == (8080, "L1", "http://127.0.0.1:8000")
    env = {"PREFIXANVIL_UPSTREAM_BASE_URL": "http://127.0.0.1:8000", "PREFIXANVIL_MODE": "L2",
           "PREFIXANVIL_EQUIV_REPORT": str(tmp_path / "missing.json")}
    with pytest.raises(ConfigError, match="PREFIXANVIL_L2_ENABLE"):
        preflight(ProxyConfig.from_env(env))
    env["PREFIXANVIL_L2_ENABLE"] = "1"
    with pytest.raises(ConfigError, match="no equivalence report"):
        preflight(ProxyConfig.from_env(env))
    assert main_from_env(env) == 2                                   # refuses to start, clear exit code
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"schema": "prefixanvil.equivalence-report/v1", "passed": True,
                                 "build_fingerprint": "sha256:old", "summary": {"L2": {"requests": 3}}}))
    env["PREFIXANVIL_EQUIV_REPORT"] = str(stale)
    with pytest.raises(ConfigError, match="different build"):
        preflight(ProxyConfig.from_env(env))


def test_l2_runs_with_passing_report(mock, tmp_path):
    from prefixanvil.equiv import run_suite
    import os
    rep = run_suite(os.path.join(os.path.dirname(__file__), "..", "fixtures", "equiv"))
    assert rep["passed"]
    p = tmp_path / "equivalence-report.json"
    p.write_text(json.dumps(rep))
    cfg = ProxyConfig(upstream=mock.url, port=0, mode="L2", l2_enable=True, equiv_report=str(p),
                      log_path=str(tmp_path / "l2.jsonl"))
    preflight(cfg)
    px = AnvilProxy(cfg).start()
    try:
        sysmsg = "Agent.\nSession clock: 2026-01-15T09:04:26Z\n\nBe precise."
        call(px.url, body=json.dumps(chat(system=sysmsg)).encode())
        fwd = json.loads(mock.received[-1]["body"])["messages"][0]["content"]
        assert fwd == "Agent.\n\nBe precise.\n\nSession clock: 2026-01-15T09:04:26Z"
        e = read_log(px)[-1]
        moved = [a for a in e["applied"] if a["rule"] == "L2.system.relocate_volatile_lines"]
        assert moved and moved[0]["lines_moved"] == 1
    finally:
        px.stop()
