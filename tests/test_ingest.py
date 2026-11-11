import json

from prefixanvil.ingest import load_jsonl, load_lines


def line(**kw):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    base.update(kw)
    return json.dumps(base)


def test_invalid_lines_are_counted_with_reasons():
    lines = [
        line(session_id="a"),
        "{not json",
        json.dumps([1, 2]),
        json.dumps({"messages": [{"role": "user", "content": "x"}]}),
        line(messages=[]),
        line(messages=[{"role": "wizard", "content": "x"}]),
        line(messages=[{"role": "user"}]),
        line(tools="nope"),
        line(timestamp="yesterday"),
        "",
        "   ",
        line(session_id="a", timestamp="2026-01-01T00:00:00Z"),
    ]
    res = load_lines(lines)
    assert len(res.requests) == 2
    assert res.blank_lines == 2
    assert res.rejected == {"invalid_json": 1, "not_an_object": 1, "missing_model": 1,
                            "empty_messages": 1, "invalid_role": 1, "missing_content": 1,
                            "invalid_tools": 1, "invalid_timestamp": 1}
    assert res.rejected_samples["invalid_json"] == [2]
    s = res.summary()
    assert s["accepted"] == 2 and s["rejected_total"] == 8 and s["total_lines"] == 12
    assert [r.line_no for r in res.requests] == [1, 12]
    assert [r.index for r in res.requests] == [0, 1]


def test_tools_optional_and_empty_tools_normalized():
    res = load_lines([line(), line(tools=[])])
    assert [r.tools for r in res.requests] == [None, None]


def test_single_turn_and_missing_session():
    res = load_lines([line()])
    r = res.requests[0]
    assert r.session_id is None and r.group_key == "nosession:m"
    assert r.timestamp is None


def test_empty_session_id_treated_as_missing():
    res = load_lines([line(session_id=""), line(session_id=7)])
    assert res.requests[0].session_id is None
    assert res.requests[1].session_id == "7"


def test_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    res = load_jsonl(str(p))
    assert res.requests == [] and res.total_lines == 0


def test_assistant_tool_call_without_content_is_valid():
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    res = load_lines([line(messages=[{"role": "user", "content": "x"}, msg])])
    assert len(res.requests) == 1


def test_timestamp_ordering():
    lines = [line(timestamp=3), line(timestamp="1970-01-01T00:00:01Z"), line()]
    res = load_lines(lines, order="timestamp")
    assert [r.line_no for r in res.requests] == [2, 1, 3]
    assert load_lines(lines).requests[0].line_no == 1


def test_content_parts_accepted():
    res = load_lines([line(messages=[{"role": "user", "content": [{"type": "text", "text": "a"}]}])])
    assert len(res.requests) == 1
