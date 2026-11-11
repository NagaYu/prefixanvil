"""Classifier tests.

Two kinds:
* hand-written pairs whose shapes deliberately differ from what AgentLoad
  generates (real-world date formats, Japanese names, JSON inside system text,
  CRLF...). These guard against the classifier only learning the generator.
* F-wise recall on AgentLoad traces with ground-truth labels. Bounds are set
  loosely BELOW the measured values (see README "Classifier evaluation").
"""

import json

import pytest

from prefixanvil.agentload import FAULTS, AgentLoadConfig, generate, write_trace
from prefixanvil.classify import (ClassifyConfig, Classifier, Item, analyze_text, analyze_tools,
                                  classify_all, evaluate, lcp)
from prefixanvil.ingest import Request, load_jsonl

C = ClassifyConfig()
BASE = ("You are a helpful assistant.\n\nGuidelines: be concise and accurate. "
        "Always cite sources when possible and explain trade-offs.")
LONG = "Retrieved: " + ("Lorem ipsum dolor sit amet consectetur adipiscing. " * 15)

SYSTEM_CASES = [
    ("F1", "Today Date: 26 Jul 2024\n\n" + BASE, "Today Date: 27 Jul 2024\n\n" + BASE),
    ("F1", "Today is Monday, March 3.\n" + BASE, "Today is Tuesday, March 4.\n" + BASE),
    ("F1", "Now: 2026-11-05 10:22:31+09:00\n" + BASE, "Now: 2026-11-05 10:29:02+09:00\n" + BASE),
    ("F1", "ts=1730800000123 " + BASE, "ts=1730800000456 " + BASE),
    ("F1", "trace_id: 4bf92f3577b34da6a3ce929d0e0e4736\n" + BASE,
           "trace_id: 00f067aa0ba902b74bf92f3577b34da6\n" + BASE),
    ("F1", "id 550e8400-e29b-41d4-a716-446655440000 " + BASE,
           "id 6fa459ea-ee8a-3ca4-894e-db77e160355e " + BASE),
    ("F2", 'Config: {"lang": "en", "tz": "UTC", "mode": "fast"}\n' + BASE,
           'Config: {"tz": "UTC", "lang": "en", "mode": "fast"}\n' + BASE),
    ("F2", 'Config: {"a": [1, 2], "b": {"c": 1}}\n' + BASE, 'Config: {"a":[1,2],"b":{"c":1}}\n' + BASE),
    ("F4", "ユーザー: 山田太郎 さん\n\n" + BASE, "ユーザー: 佐藤花子 さん\n\n" + BASE),
    ("F4", "Contact: alice@example.com\n\n" + BASE, "Contact: bob.smith@corp.io\n\n" + BASE),
    ("F4", "You work for Acme Corp.\n\n" + BASE, "You work for Globex Corporation.\n\n" + BASE),
    ("F5", "Intro.\n\n## Beta features\nYou may use beta tools.\n\n" + BASE, "Intro.\n\n" + BASE),
    ("F5", "Intro.\n\n" + BASE, "Intro.\n\n## Safety\nRefuse unsafe requests.\n\n" + BASE),
    ("F5", "Intro.\nNote: debug mode enabled.\n" + BASE, "Intro.\n" + BASE),
    ("F6", BASE.replace("\n", "\r\n"), BASE),
    ("F6", "Intro:\tx\n" + BASE, "Intro: x\n" + BASE),
    ("F6", BASE, BASE + "\n"),
    ("F7", "Intro.\n\n" + BASE, "Intro.\n\n" + LONG + "\n\n" + BASE),
    ("F7", "Intro.\n\n" + LONG + "\n\n" + BASE, "Intro.\n\n" + LONG[::-1] + "\n\n" + BASE),
    ("unknown", "You are a helpful assistant who answers questions.\n\n" + BASE,
                "Act as a senior engineer reviewing pull requests.\n\n" + BASE),
    # known limitation: non-JSON key/value reorder is not recognised as F2
    ("unknown", "lang: en\ntz: UTC\n\n" + BASE, "tz: UTC\nlang: en\n\n" + BASE),
]


@pytest.mark.parametrize("expected,a,b", SYSTEM_CASES)
def test_system_text_cases(expected, a, b):
    label, _ = analyze_text(a, b, C, system=True)
    assert label == expected


def test_non_system_text_falls_through():
    # past user turn edited -> analyze_text gives None, caller labels F8
    assert analyze_text("please fix the bug in parser", "please fix the parser", C, system=False) is None
    assert analyze_text('{"a": 1, "b": 2}', '{"b": 2, "a": 1}', C, system=False)[0] == "F2"


def T(name, **fn):
    return {"type": "function", "function": {"name": name, "description": "d", **fn}}


def test_tools_cases():
    a, b, c = T("a"), T("b"), T("c")
    assert analyze_tools([a, b], [b, a], C) == ("F3", "tools_order_only")
    assert analyze_tools([a, b], [a, b, c], C)[0] == "F3"
    assert analyze_tools([a, b], [a], C)[0] == "F3"
    reordered = {"function": {"description": "d", "name": "a"}, "type": "function"}
    assert analyze_tools([a], [reordered], C) == ("F2", "tool_schema_key_order")
    assert analyze_tools([T("a", description="x y")], [T("a", description="x  y")], C)[0] == "F6"
    assert analyze_tools([a], [T("a", parameters={"type": "object"})], C)[0] == "unknown"


# --------------------------------------------------------------------------- #
# full pair classification through the fixture renderer
# --------------------------------------------------------------------------- #

def item(renderer, messages, tools=None, idx=0, sid="s"):
    r = Request(idx, idx + 1, "m", messages, tools, sid, None, f"r{idx}")
    return Item(r, renderer.render(r))


SYS = {"role": "system", "content": "You are an agent.\n\n" + BASE}
U1 = {"role": "user", "content": "List the files in the repo please."}
A1 = {"role": "assistant", "content": None, "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": '{"path": ".", "depth": 1}'}}]}
R1 = {"role": "tool", "tool_call_id": "c1", "content": "README.md\nsrc/\ntests/\n" * 5}


def test_pair_extension_is_none(renderer):
    clf = Classifier(renderer)
    p = item(renderer, [SYS, U1])
    c = item(renderer, [SYS, U1, A1, R1], idx=1)
    r = clf.classify_pair(p, c, "within_session")
    assert r.label == "none" and r.lost_tokens == 0 and r.first_divergence_token is None


def test_pair_history_rewrite_and_loss(renderer):
    clf = Classifier(renderer)
    p = item(renderer, [SYS, U1, A1, R1])
    r1_short = dict(R1, content="README.md ...[truncated]")
    c = item(renderer, [SYS, U1, A1, r1_short, {"role": "user", "content": "thanks"}], idx=1)
    r = clf.classify_pair(p, c, "within_session")
    assert r.label == "F8" and r.element == "messages[3]"
    ptoks, ctoks = p.rendered.token_ids, c.rendered.token_ids
    assert r.first_divergence_token == lcp(ptoks, ctoks)
    assert r.actual_shared_tokens == (lcp(ptoks, ctoks) // 16) * 16
    assert r.lost_tokens == r.expected_shared_tokens - r.actual_shared_tokens > 0


def test_pair_tool_args_reserialized_is_f2(renderer):
    clf = Classifier(renderer)
    a1b = json.loads(json.dumps(A1))
    a1b["tool_calls"][0]["function"]["arguments"] = '{"depth": 1, "path": "."}'
    p = item(renderer, [SYS, U1, A1, R1])
    c = item(renderer, [SYS, U1, a1b, R1, {"role": "user", "content": "ok"}], idx=1)
    assert clf.classify_pair(p, c, "within_session").label == "F2"


def test_pair_tools_before_messages(renderer):
    clf = Classifier(renderer)
    p = item(renderer, [SYS, U1], [T("a"), T("b")])
    c = item(renderer, [SYS, U1, A1, R1], [T("b"), T("a")], idx=1)
    r = clf.classify_pair(p, c, "within_session")
    assert (r.label, r.element) == ("F3", "tools")


def test_cross_session_new_conversation_is_none(renderer):
    clf = Classifier(renderer)
    p = item(renderer, [SYS, U1], sid="a")
    c = item(renderer, [SYS, {"role": "user", "content": "Different task entirely."}], idx=1, sid="b")
    r = clf.classify_pair(p, c, "cross_session")
    assert r.label == "none" and r.lost_tokens == 0


def test_cross_session_volatile_system(renderer):
    clf = Classifier(renderer)
    s1 = {"role": "system", "content": "Clock: 2026-01-01T00:00:00Z\n\n" + BASE}
    s2 = {"role": "system", "content": "Clock: 2026-01-01T00:05:00Z\n\n" + BASE}
    p = item(renderer, [s1, U1], sid="a")
    c = item(renderer, [s2, {"role": "user", "content": "other"}], idx=1, sid="b")
    r = clf.classify_pair(p, c, "cross_session")
    assert r.label == "F1" and r.lost_tokens > 0


def test_template_rerender_is_f6(renderer):
    class Rewriting:
        """Pretend template that drops content of earlier assistant turns."""
        def __init__(self, inner):
            self.inner = inner
            self.name = inner.name

        def template_info(self):
            return self.inner.template_info()

        def render(self, messages, tools, add_generation_prompt):
            msgs = [dict(m, content="<hidden>") if m["role"] == "assistant" and i < len(messages) - 2
                    else m for i, m in enumerate(messages)]
            return self.inner.render(msgs, tools, add_generation_prompt)

        def encode(self, text):
            return self.inner.encode(text)

    from prefixanvil.render import Renderer
    rr = Renderer(Rewriting(renderer.tok))
    clf = Classifier(rr)
    a = {"role": "assistant", "content": "Here is the answer."}
    p = item(rr, [SYS, U1, a, {"role": "user", "content": "more"}])
    c = item(rr, [SYS, U1, a, {"role": "user", "content": "more"}, a, {"role": "user", "content": "x"}], idx=1)
    r = clf.classify_pair(p, c, "within_session")
    assert (r.label, r.detail) == ("F6", "template_rerenders_history")


def test_nosession_pairing_uses_best_match(renderer, tmp_path):
    lines = [
        {"model": "m", "messages": [SYS, U1]},
        {"model": "m", "messages": [{"role": "system", "content": "Other bot."}, U1]},
        {"model": "m", "messages": [SYS, U1, A1, R1]},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    reqs = load_jsonl(str(p)).requests
    items = [Item(r, renderer.render(r)) for r in reqs]
    res = classify_all(items, Classifier(renderer))
    last = res[-1]
    assert last.reference_index == 0 and last.scope == "within_session" and last.label == "none"


# --------------------------------------------------------------------------- #
# F-wise recall on AgentLoad ground truth
# --------------------------------------------------------------------------- #

# Loose lower bounds, deliberately below the measured values (README documents
# the measured numbers). Do not tighten to 1.0: the bound should catch
# regressions, not freeze the current rule set.
RECALL_FLOOR = {f: 0.8 for f in FAULTS}
PRECISION_FLOOR = 0.8


@pytest.fixture(scope="module")
def evaluation(renderer, tmp_path_factory):
    from prefixanvil.pipeline import AnalyzeOptions, analyze
    d = tmp_path_factory.mktemp("eval")
    agg = {f: [0, 0, 0] for f in FAULTS}
    for seed, (sys_tokens, tools) in enumerate([(300, 6), (900, 2), (1500, 12)]):
        cfg = AgentLoadConfig.from_dict({"seed": 100 + seed, "sessions": 30, "turns": 2,
                                         "system_tokens": sys_tokens, "tools": tools,
                                         "retry_rate": 0.1, "faults": {f: 0.1 for f in FAULTS}})
        reqs, labels = generate(cfg)
        p = d / f"t{seed}.jsonl"
        lp = write_trace(reqs, labels, str(p))
        ev = analyze(str(p), AnalyzeOptions(labels=lp))["classifier_evaluation"]
        for f, v in ev["per_fault"].items():
            agg[f][0] += v["tp"]
            agg[f][1] += v["support"]
            agg[f][2] += v["predicted"]
    return agg


@pytest.mark.parametrize("fault", FAULTS)
def test_recall_floor_per_fault(evaluation, fault):
    tp, support, predicted = evaluation[fault]
    assert support >= 5, f"too few {fault} samples to evaluate"
    assert tp / support >= RECALL_FLOOR[fault]
    assert tp / predicted >= PRECISION_FLOOR
