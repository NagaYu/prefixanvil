"""Core API: versioned JSON in/out, caller-owned session context, fail-open, L2 transform."""

import json

from prefixanvil.core import TRANSFORM_REQUEST_SCHEMA, transform
from prefixanvil.relocate import is_volatile_only_line, relocate_text


def T(name):
    return {"type": "function", "function": {"name": name}}


def payload(body, mode="L1", ctx="new"):
    if ctx == "new":
        ctx = {"schema": "prefixanvil.session-context/v1", "prev_tool_names": None, "prev_tool_order": None}
    return {"schema": TRANSFORM_REQUEST_SCHEMA, "mode": mode, "request": body, "session_context": ctx}


BODY = {"model": "m", "temperature": 0, "stream": True,
        "messages": [{"role": "system", "content": "Hi  there.\nRequest id: 550e8400-e29b-41d4-a716-446655440000\n\nRules."},
                     {"role": "user", "content": "x"}],
        "tools": [T("b"), T("a")]}


def test_result_schema_and_purity():
    p = payload(BODY)
    snapshot = json.dumps(p)
    r1, r2 = transform(p), transform(p)
    assert r1 == r2 and json.dumps(p) == snapshot                    # deterministic, input untouched
    assert r1["schema"] == "prefixanvil.transform.result/v1"
    assert r1["request"]["stream"] is True and r1["request"]["temperature"] == 0   # other fields kept
    assert r1["session_context"]["prev_tool_order"] == ["b", "a"]
    json.dumps(r1)                                                    # JSON-serialisable


def test_l0_reports_would_apply_but_returns_original():
    r = transform(payload(BODY, "L0"))
    assert r["request"] is BODY and r["changed"] is False
    assert {w["rule"] for w in r["would_apply"]} >= {"L1.system.whitespace"}


def test_l2_relocates_then_l1():
    r = transform(payload(BODY, "L2"))
    sysc = r["request"]["messages"][0]["content"]
    assert sysc == "Hi there.\n\nRules.\n\nRequest id: 550e8400-e29b-41d4-a716-446655440000"
    assert [a["rule"] for a in r["applied"]][0] == "L2.system.relocate_volatile_lines"
    assert transform(payload(BODY, "L1"))["request"]["messages"][0]["content"].startswith("Hi there.\nRequest id")


def test_fail_open_cases():
    for bad, reason in [({"schema": "x/v9"}, "unsupported_schema"),
                        (payload(BODY, "L7"), "unsupported_mode"),
                        (payload({"model": "m", "messages": []}), "invalid_request:empty_messages"),
                        (payload(BODY, ctx={"schema": "other/v1"}), "internal_error:ValueError")]:
        r = transform(bad)
        assert r["skipped"][0]["reason"] == reason and r["changed"] is False
        assert r["request"] is (bad.get("request") if isinstance(bad, dict) else None)


def test_canonicalizer_exception_is_fail_open():
    def boom(req, ctx):
        raise RuntimeError("x")
    r = transform(payload(BODY), _canonicalize=boom)
    assert r["request"] is BODY and r["skipped"][0]["reason"] == "internal_error:RuntimeError"


def test_volatile_only_line_detector():
    yes = ["Session clock: 2026-01-15T09:04:26Z", "Request id: 550e8400-e29b-41d4-a716-446655440000",
           "Run: run-000123", "Epoch: 1768467866", "Today Date: 26 Jul 2024", "2026-01-01",
           "- trace_id = 4bf92f3577b34da6a3ce929d0e0e4736"]
    no = ["Use at most 500 tokens.", "Compliance: RFC-2119", "Release date: 2024-07-26 (stable)",
          "Today is Monday, March 3.", "Ratio 16:9 preferred.", "", "Generated at 2026-01-01 10:00:00Z by bot"]
    assert all(is_volatile_only_line(x) for x in yes)
    assert not any(is_volatile_only_line(x) for x in no)


def test_relocate_text_keeps_order_and_fences():
    t = "A\nTs: 2026-01-01T00:00:00Z\nB\nId: 550e8400-e29b-41d4-a716-446655440000\n```\nTs: 2026-01-01T00:00:00Z\n```"
    out, n = relocate_text(t)
    assert n == 2
    assert out == ("A\nB\n```\nTs: 2026-01-01T00:00:00Z\n```\n\n"
                   "Ts: 2026-01-01T00:00:00Z\nId: 550e8400-e29b-41d4-a716-446655440000")
    assert relocate_text("nothing here") == ("nothing here", 0)
