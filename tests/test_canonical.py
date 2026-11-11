import json

from prefixanvil.canonical import canonicalize, normalize_system_whitespace
from prefixanvil.ingest import Request


def mk(messages, tools=None):
    return Request(0, 1, "m", messages, tools, "s", None, None)


def tool(name, **extra):
    fn = {"parameters": {"type": "object", "required": ["b", "a"],
                         "properties": {"b": {"type": "string"}, "a": {"type": "integer"}}},
          "name": name, "description": "d"}
    fn.update(extra)
    return {"function": fn, "type": "function"}


MSGS = [
    {"role": "system", "content": "Hello  world.   Bye \n\n\n\n  indented  line\n\n"},
    {"role": "user", "content": "do  it"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "zeta", "arguments": '{"y": 1,   "x": [3, 1]}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"b":2,"a":{"d":1,"c":0}}'},
]
TOOLS = [tool("zeta"), tool("alpha")]


def test_rules_apply_and_are_logged():
    req = mk(MSGS, TOOLS)
    out, log = canonicalize(req)
    rules = {e["rule"] for e in log.applied}
    assert rules == {"L1.tools.sort_keys", "L1.tools.sort_by_name",
                     "L1.json_string.normalize", "L1.system.whitespace"}
    for e in log.applied:
        assert len(e["before_sha256"]) == 16 and e["before_sha256"] != e["after_sha256"]
    assert [t["function"]["name"] for t in out.tools] == ["alpha", "zeta"]
    assert list(out.tools[0]) == ["function", "type"]
    assert list(out.tools[0]["function"]["parameters"]["properties"]) == ["a", "b"]
    assert out.tools[0]["function"]["parameters"]["required"] == ["b", "a"]   # arrays untouched
    assert out.messages[0]["content"] == "Hello world. Bye\n\n  indented line"
    assert out.messages[1]["content"] == "do  it"                              # user text untouched
    assert out.messages[2]["tool_calls"][0]["function"]["arguments"] == '{"x": [3, 1], "y": 1}'
    assert out.messages[3]["content"] == '{"a": {"c": 0, "d": 1}, "b": 2}'
    # input not mutated
    assert req.messages[0]["content"].startswith("Hello  world")
    assert req.tools[0]["function"]["name"] == "zeta"


def test_idempotent():
    once, _ = canonicalize(mk(MSGS, TOOLS))
    twice, log2 = canonicalize(once)
    assert twice.messages == once.messages and twice.tools == once.tools
    assert json.dumps(twice.tools) == json.dumps(once.tools)
    assert log2.applied == []


def test_json_equivalence_preserved():
    out, _ = canonicalize(mk(MSGS, TOOLS))
    by_name = {t["function"]["name"]: t for t in TOOLS}
    for t in out.tools:
        assert t == by_name[t["function"]["name"]]            # dict equality ignores key order
    a0 = MSGS[2]["tool_calls"][0]["function"]["arguments"]
    a1 = out.messages[2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(a0) == json.loads(a1)
    assert json.loads(MSGS[3]["content"]) == json.loads(out.messages[3]["content"])


def test_fail_open_shapes():
    weird_tools = [{"type": "function", "function": {"description": "no name"}}, tool("b")]
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "parts  list"}]},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "f", "arguments": "not json {"}},
            {"id": "2", "function": {"name": "f", "arguments": "42"}},
            {"id": "3", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "plain text  result"},
    ]
    out, log = canonicalize(mk(msgs, weird_tools))
    reasons = {(e["rule"], e["reason"]) for e in log.skipped}
    assert ("L1.tools.sort_by_name", "tool_without_name") in reasons
    assert ("L1.system.whitespace", "non_string_content") in reasons
    assert ("L1.json_string.normalize", "not_json") in reasons
    assert ("L1.json_string.normalize", "json_scalar") in reasons
    assert ("L1.json_string.normalize", "unexpected_shape") in reasons
    assert [t["function"].get("name") for t in out.tools] == [None, "b"]     # order kept
    assert out.messages[0]["content"] == msgs[0]["content"]
    assert out.messages[2]["tool_calls"][0]["function"]["arguments"] == "not json {"
    assert out.messages[3]["content"] == "plain text  result"


def test_duplicate_tool_names_not_sorted():
    out, log = canonicalize(mk([{"role": "user", "content": "x"}], [tool("b"), tool("a"), tool("b")]))
    assert [t["function"]["name"] for t in out.tools] == ["b", "a", "b"]
    assert any(e["reason"] == "duplicate_tool_names" for e in log.skipped)


def test_no_tools_no_changes():
    out, log = canonicalize(mk([{"role": "user", "content": "hi"}]))
    assert out.tools is None and log.applied == [] and log.skipped == []


def test_whitespace_normalizer_cases():
    f = normalize_system_whitespace
    assert f("a\n") == "a"
    assert f("a  \nb\t\n") == "a\nb"
    assert f("a\n\n\n\nb") == "a\n\nb"
    assert f("  two  spaces") == "  two spaces"         # leading indentation kept
    assert f("    code  block") == "    code  block"    # Markdown indented code untouched (Phase 2)
    fenced = "Intro.  \n```py\nx  = 1   \n\n\n\ny = 2\n```\ntail  \n\n"
    assert f(fenced) == "Intro.\n```py\nx  = 1   \n\n\n\ny = 2\n```\ntail"
    assert f("open fence\n```\na  b\n") == "open fence\n```\na  b\n"   # unclosed fence protects the rest
    for s in ("x", "a  b\n\n\nc  \n", "\t\tq  r", fenced, "```\n  a  \n"):
        assert f(f(s)) == f(s)
