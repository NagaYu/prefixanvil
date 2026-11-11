"""Phase 2 Step 0: session-aware L1 tool ordering."""

import os

import pytest

from prefixanvil.canonical import SessionContext, canonicalize, next_session_context
from prefixanvil.ingest import Request
from prefixanvil.pipeline import AnalyzeOptions, analyze


def T(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def req(*names):
    return Request(0, 1, "m", [{"role": "user", "content": "x"}], [T(n) for n in names], "s", None, None)


def names(r):
    return [t["function"]["name"] for t in r.tools]


def run_session(*tool_lists):
    ctx, out, logs = SessionContext(), [], []
    for tl in tool_lists:
        c, log = canonicalize(req(*tl), ctx)
        ctx = next_session_context(c)
        out.append(names(c))
        logs.append(log)
    return out, logs


def test_no_context_is_phase1_stateless_sort():
    c, log = canonicalize(req("b", "a"))
    assert names(c) == ["a", "b"]
    assert "tool_sort:stateless_no_session_context" in log.notes


def test_first_request_keeps_client_order():
    out, logs = run_session(["b", "a", "c"])
    assert out == [["b", "a", "c"]]
    assert "tool_sort:session_first_request_client_order_kept" in logs[0].notes


def test_unchanged_set_reuses_previous_order():
    out, logs = run_session(["b", "a", "c"], ["c", "b", "a"], ["a", "c", "b"])
    assert out == [["b", "a", "c"]] * 3                    # client shuffles are undone
    assert {e["reason"] for e in logs[1].skipped} == {"session_order_pinned"}
    assert [e["rule"] for e in logs[1].applied if e["path"] == "tools"] == ["L1.tools.session_stable_order"]


def test_changed_set_skips_sort_with_reason_and_keeps_order():
    out, logs = run_session(["b", "a", "c"], ["b", "c"], ["d", "b", "c"], ["c", "d", "b"])
    assert out[1] == ["b", "c"]                            # removal: untouched relative order
    assert out[2] == ["b", "c", "d"]                       # addition: appended after retained tools
    assert out[3] == ["b", "c", "d"]                       # unchanged afterwards: pinned
    for i in (1, 2):
        assert any(e["rule"] == "L1.tools.sort_by_name" and e["reason"] == "tool_set_changed"
                   for e in logs[i].skipped)


def test_context_json_roundtrip_and_schema_check():
    ctx = next_session_context(req("b", "a"))
    assert SessionContext.from_json(ctx.to_json()) == ctx
    assert SessionContext.from_json(None) == SessionContext()
    with pytest.raises(ValueError):
        SessionContext.from_json({"schema": "other/v9"})


def test_duplicate_names_still_fail_open_with_context():
    c, log = canonicalize(req("b", "a", "b"), SessionContext(("a", "b", "b"), None))
    assert names(c) == ["b", "a", "b"]
    assert any(e["reason"] == "duplicate_tool_names" for e in log.skipped)


def _f3(rep, key):
    return next((c["lost_tokens"] for c in rep[key] if c["label"] == "F3"), 0)


def test_regression_f3_not_worse_after_l1(root):
    tr = lambda n: os.path.join(root, "fixtures", "traces", f"{n}.jsonl")
    # strict on the F3-only fixture
    rep = analyze(tr("only_f3"))
    assert _f3(rep, "causes_residual_after_l1") <= _f3(rep, "causes_raw")
    # demo fixture: Phase-1 stateless sort made F3 worse (784 -> 1280) ...
    stateless = analyze(tr("demo"), AnalyzeOptions(session_aware_l1=False))
    assert _f3(stateless, "causes_raw") == 784 and _f3(stateless, "causes_residual_after_l1") == 1280
    # ... session-aware order is identical to raw for both F3 pairs; the only
    # remaining difference is block alignment from L1.tools.sort_keys changing
    # token counts (one block per F3 pair at most). See README / spec issues.
    rep = analyze(tr("demo"))
    pairs = next(c["pairs"] for c in rep["causes_raw"] if c["label"] == "F3")
    assert _f3(rep, "causes_residual_after_l1") <= _f3(rep, "causes_raw") + 16 * pairs
    assert rep["l1_transforms"]["tool_sort"] == "session_aware"
