"""Equivalence gate: mock semantics, fixture suite, and a deliberately bad canonicalizer."""

import copy
import json
import os

import pytest

from prefixanvil.canonical import canonicalize
from prefixanvil.cli import main as prefixanvil
from prefixanvil.equiv import REPORT_SCHEMA, build_fingerprint, check_report, load_suite, run_suite
from prefixanvil.mock_upstream import mock_completion_text

SUITE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "equiv")


def body(**kw):
    b = {"model": "m", "messages": [{"role": "system", "content": "Be brief.\n```\nx  = 1\n```"},
                                    {"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
                   {"type": "function", "function": {"name": "b"}}]}
    b.update(kw)
    return b


def test_mock_is_deterministic_and_semantic():
    b = body()
    assert mock_completion_text(b) == mock_completion_text(copy.deepcopy(b))
    # pure re-serialization / declared non-semantic changes -> same output
    same = copy.deepcopy(b)
    same["tools"] = [{"function": {"parameters": {"type": "object"}, "name": "a"}, "type": "function"},
                     {"type": "function", "function": {"name": "b"}}][::-1]
    same["messages"][0]["content"] = "Be  brief.  \n```\nx  = 1\n```\n\n"
    same["stream"] = True
    assert mock_completion_text(same) == mock_completion_text(b)
    # semantic changes -> different output, first differing word points at the element
    for mutate in (lambda d: d["messages"][1].update(content="hi!"),
                   lambda d: d["messages"][0].update(content="Be brief.\n```\nx = 1\n```"),
                   lambda d: d["tools"].pop(),
                   lambda d: d.update(temperature=0.7)):
        d = copy.deepcopy(b)
        mutate(d)
        assert mock_completion_text(d) != mock_completion_text(b)


def test_fixture_suite_passes():
    cases, _ = load_suite(SUITE)
    assert len(cases) >= 15
    ids = {c["id"] for c in cases}
    assert {"ws_code_block", "duplicate_tool_names", "tool_set_change_mid_session",
            "nested_json_string_argument", "non_json_tool_content"} <= ids
    rep = run_suite(SUITE)
    assert rep["schema"] == REPORT_SCHEMA and rep["passed"], rep["unexpected_mismatches"]
    assert rep["summary"]["L1"]["exact_match_rate"] == 1.0
    assert [a["case"] for a in rep["allowed_diffs"]] == ["l2_volatile_lines"]
    assert run_suite(SUITE)["results"] == rep["results"]             # deterministic


def _bad_dedent(req, ctx):
    """Semantics-changing 'canonicalizer': strips indentation everywhere in system text."""
    new, log = canonicalize(req, ctx)
    for m in new.messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = "\n".join(line.strip() for line in m["content"].split("\n"))
    return new, log


def _bad_nested_json(req, ctx):
    """Naive recursive JSON re-serialization that also rewrites JSON *inside* string values."""
    new, log = canonicalize(req, ctx)
    for m in new.messages:
        for tc in m.get("tool_calls") or []:
            args = tc["function"].get("arguments")
            if isinstance(args, str):
                obj = json.loads(args)
                for k, v in obj.items():
                    if isinstance(v, str) and v.startswith("{"):
                        obj[k] = json.dumps(json.loads(v), sort_keys=True)
                tc["function"]["arguments"] = json.dumps(obj, sort_keys=True)
    return new, log


@pytest.mark.parametrize("bad,case", [(_bad_dedent, "ws_code_block"),
                                      (_bad_nested_json, "nested_json_string_argument")])
def test_bad_canonicalizer_fails_the_gate(bad, case):
    rep = run_suite(SUITE, canonicalizer=bad)
    assert not rep["passed"]
    assert case in {u["case"] for u in rep["unexpected_mismatches"]}


def test_check_report_and_cli(tmp_path, capsys):
    out = tmp_path / "equivalence-report.json"
    assert prefixanvil(["equiv", "--suite", SUITE, "--out", str(out)]) == 0
    assert (tmp_path / "equivalence-report.md").exists()
    assert "PASSED" in capsys.readouterr().out
    assert check_report(str(out)) == (True, "ok")
    rep = json.loads(out.read_text())
    assert rep["build_fingerprint"] == build_fingerprint()
    rep["passed"] = False
    out.write_text(json.dumps(rep))
    assert check_report(str(out))[0] is False
    assert check_report(str(tmp_path / "nope.json"))[0] is False


def test_suite_validation(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"schema": "prefixanvil.equiv-case/v1", "id": "x",
                                                 "requests": [body()], "expect": {"L1": "allowed-diff"}}))
    with pytest.raises(ValueError, match="allowed_diff_reason"):
        load_suite(str(tmp_path))
