"""AgentLoad: synthetic agent-style trace generator with labelled fragmentation.

Design (see README "AgentLoad"):

* Each session is a sequence of requests produced by an agent loop:
  user turn -> N tool-call steps (the "burst") -> final assistant answer.
  Request k+1 of a session is, absent faults, an append-only extension of
  request k (it adds the assistant reply and tool results).
* Fragmentation is injected as *events*. Between a request and its reference
  (the previous request of the same session, or for a session's first request
  the previous session's first request) at most ONE event is applied, so the
  ground-truth cause of the first divergence is unambiguous.
* F4 (per-user interpolation) is only injected across sessions, F8 (history
  rewrite) only within a session. All other F's can occur in both scopes.
* Everything is driven by a single ``random.Random(seed)``; output bytes are a
  pure function of the config.
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any

FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
CROSS_SESSION_FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
WITHIN_SESSION_FAULTS = ("F1", "F2", "F3", "F5", "F6", "F7", "F8")

_WORDS = (
    "agent build cache check code commit config context data debug deploy "
    "detail error event file fix function graph handle index input issue job "
    "kernel layer limit log memory merge metric model module network node "
    "output package parse patch path plan policy process project query queue "
    "record release repo request result review route rule run schema search "
    "server service shard source state step store stream task test thread "
    "token tool trace update user value version worker workflow always "
    "carefully clearly never only first then before after when while should "
    "must may can will the a an of to in for with on by from and or not "
    "each every any this that these those your our their its use keep make "
    "read write list find open close return report explain verify ensure"
).split()

_FIRST_NAMES = ("Alice", "Bruno", "Chiara", "Deepak", "Elena", "Farid", "Grace",
                "Hiroshi", "Ingrid", "Jamal", "Keiko", "Lucas", "Mei", "Nadia",
                "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tomoko")
_LAST_NAMES = ("Andersen", "Barros", "Costa", "Dubois", "Eriksen", "Fujita",
               "Garcia", "Hansen", "Ivanova", "Jensen", "Kowalski", "Larsen",
               "Moreau", "Nakamura", "Okafor", "Petrov", "Quinn", "Rossi")
_TEAMS = ("platform", "payments", "search", "infra", "mobile", "data")
_VERBS = ("read", "write", "search", "list", "run", "fetch", "update", "delete",
          "create", "inspect", "diff", "lint", "build", "deploy", "query")
_NOUNS = ("file", "directory", "tests", "issue", "branch", "logs", "metrics",
          "config", "schema", "ticket", "docs", "package", "service", "table")
_VOLATILE_KINDS = ("iso_time", "uuid", "counter", "epoch")

_BASE_TIME = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)


@dataclass
class AgentLoadConfig:
    seed: int = 0
    model: str = "agent-model"
    sessions: int = 8
    turns: int = 3
    system_tokens: int = 600
    tools: int = 8
    tool_burst_min: int = 1
    tool_burst_max: int = 3
    retry_rate: float = 0.05
    faults: dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FAULTS})
    chars_per_token: float = 4.0
    session_gap_s: float = 20.0
    step_gap_s: float = 4.0
    # Client behaviour (not a labelled fault): "fixed" = every session lists tools in
    # the same order; "per_session_shuffle" = each session's client uses its own
    # (seeded) tool order, constant within the session. Used by bench_xsession.
    client_tool_order: str = "fixed"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentLoadConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown AgentLoad config keys: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in d.items() if k != "faults"})
        faults = {f: 0.0 for f in FAULTS}
        for k, v in (d.get("faults") or {}).items():
            if k not in FAULTS:
                raise ValueError(f"unknown fault id {k!r}; expected one of {FAULTS}")
            faults[k] = float(v)
        cfg.faults = faults
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.sessions < 0 or self.turns < 1 or self.tools < 0:
            raise ValueError("sessions>=0, turns>=1, tools>=0 required")
        if not (0 <= self.tool_burst_min <= self.tool_burst_max):
            raise ValueError("0 <= tool_burst_min <= tool_burst_max required")
        if self.client_tool_order not in ("fixed", "per_session_shuffle"):
            raise ValueError("client_tool_order must be 'fixed' or 'per_session_shuffle'")
        if not 0.0 <= self.retry_rate <= 1.0:
            raise ValueError("retry_rate must be in [0,1]")
        for f, p in self.faults.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"fault probability {f}={p} outside [0,1]")
        for scope in (CROSS_SESSION_FAULTS, WITHIN_SESSION_FAULTS):
            s = sum(self.faults[f] for f in scope)
            if s > 1.0 + 1e-9:
                raise ValueError(
                    f"fault probabilities for one scope sum to {s:.3f} > 1 "
                    "(at most one event is injected per request)")


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def _sentence(rng: random.Random, n_min: int = 6, n_max: int = 14) -> str:
    words = [rng.choice(_WORDS) for _ in range(rng.randint(n_min, n_max))]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _paragraph(rng: random.Random, target_chars: int) -> str:
    out: list[str] = []
    while sum(len(s) + 1 for s in out) < target_chars:
        out.append(_sentence(rng))
    return " ".join(out)


def _volatile_value(kind: str, rng: random.Random, state: dict) -> str:
    if kind == "iso_time":
        state["clock"] = state.get("clock", 0) + rng.randint(5, 600)
        t = _BASE_TIME + timedelta(seconds=state["clock"])
        return "Session clock: " + t.strftime("%Y-%m-%dT%H:%M:%SZ")
    if kind == "uuid":
        return "Request id: " + str(uuid.UUID(int=rng.getrandbits(128), version=4))
    if kind == "counter":
        state["counter"] = state.get("counter", rng.randint(100, 9000)) + 1
        return f"Run: run-{state['counter']:06d}"
    if kind == "epoch":
        state["clock"] = state.get("clock", 0) + rng.randint(5, 600)
        return f"Epoch: {int(_BASE_TIME.timestamp()) + state['clock']}"
    raise ValueError(kind)


def _make_tool(rng: random.Random, name: str) -> dict:
    n_params = rng.randint(2, 4)
    props: dict[str, Any] = {}
    for _ in range(n_params):
        p = rng.choice(_NOUNS) + "_" + rng.choice(("id", "path", "name", "limit", "query"))
        props[p] = {"type": "integer" if p.endswith("limit") else "string",
                    "description": _sentence(rng, 4, 8)}
    keys = list(props)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _sentence(rng, 8, 16),
            "parameters": {"type": "object", "properties": props,
                           "required": keys[: max(1, len(keys) // 2)]},
        },
    }


def _tool_name(t: dict) -> str:
    return t["function"]["name"]


def _shuffle_dict_keys(d: dict, rng: random.Random) -> dict:
    keys = list(d)
    if len(keys) < 2:
        return d
    new = keys[:]
    while new == keys:
        rng.shuffle(new)
    return {k: d[k] for k in new}


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #

@dataclass
class _SystemState:
    sections: list[list[Any]]          # [key, text, enabled]
    joiners: list[str]                  # separator after section i (len = len(sections)-1)
    trailing: str
    volatile_kind: str
    volatile_state: dict

    def render(self) -> str:
        parts: list[str] = []
        for i, (key, text, enabled) in enumerate(self.sections):
            if not enabled:
                continue
            if parts:
                parts.append(self.joiners[i - 1] if i > 0 else "\n\n")
            parts.append(text)
        return "".join(parts) + self.trailing

    def idx(self, key: str) -> int:
        for i, s in enumerate(self.sections):
            if s[0] == key:
                return i
        raise KeyError(key)


def _initial_system(cfg: AgentLoadConfig, rng: random.Random) -> _SystemState:
    total = int(cfg.system_tokens * cfg.chars_per_token)
    kind = rng.choice(_VOLATILE_KINDS)
    vstate: dict = {}
    sections: list[list[Any]] = [
        ["header", "You are Forge Agent, an autonomous software engineering assistant.", True],
        ["volatile", _volatile_value(kind, rng, vstate), True],
        ["profile", _profile(rng), True],
        ["identity", _paragraph(rng, max(80, total // 5)), True],
    ]
    for i in range(3):
        sections.append([f"optional{i}",
                         f"## Conditional note {i + 1}\n" + _paragraph(rng, rng.randint(120, 330)),
                         rng.random() < 0.5])
    sections.append(["rag", _rag_block(rng), True])
    used = sum(len(s[1]) for s in sections)
    remaining = max(200, total - used)
    n_guides = 3
    for i in range(n_guides):
        sections.append([f"guideline{i}",
                         f"## Guideline {i + 1}\n" + _paragraph(rng, remaining // n_guides), True])
    return _SystemState(sections, ["\n\n"] * (len(sections) - 1), "", kind, vstate)


def _profile(rng: random.Random) -> str:
    return (f"You are assisting {rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)} "
            f"from the {rng.choice(_TEAMS)} team.")


def _rag_block(rng: random.Random) -> str:
    docs = []
    for i in range(rng.randint(2, 4)):
        docs.append(f"[doc {i + 1}] {rng.choice(_NOUNS)}/{rng.choice(_NOUNS)}.md: "
                    + _paragraph(rng, rng.randint(250, 450)))
    return "## Retrieved context\n" + "\n".join(docs)


@dataclass
class _SessionState:
    system: _SystemState
    tools: list[dict]
    tool_pool: list[dict]             # tools that may be added by F3
    history: list[dict] = field(default_factory=list)

    def snapshot(self) -> tuple[list[dict], list[dict]]:
        msgs = [{"role": "system", "content": self.system.render()}] + copy.deepcopy(self.history)
        return msgs, copy.deepcopy(self.tools)


# --------------------------------------------------------------------------- #
# fault events: each returns a location dict, or None if not applicable
# --------------------------------------------------------------------------- #

def _ev_f1(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    s = st.system
    i = s.idx("volatile")
    s.sections[i][1] = _volatile_value(s.volatile_kind, rng, s.volatile_state)
    return {"element": "system", "section": "volatile", "detail": s.volatile_kind}


def _ev_f2(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    hist_cands = [i for i, m in enumerate(st.history[:p_len])
                  if m.get("tool_calls") and any(
                      len(json.loads(tc["function"]["arguments"])) >= 2 for tc in m["tool_calls"])]
    if hist_cands and (not st.tools or rng.random() < 0.5):
        i = rng.choice(hist_cands)
        m = st.history[i]
        for tc in m["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            if len(args) >= 2:
                tc["function"]["arguments"] = json.dumps(_shuffle_dict_keys(args, rng))
                break
        return {"element": f"messages[{i + 1}]", "detail": "tool_call_arguments_key_order"}
    if not st.tools:
        return None
    ti = rng.randrange(len(st.tools))
    t = st.tools[ti]
    if rng.random() < 0.5:
        t["function"] = _shuffle_dict_keys(t["function"], rng)
        where = "function"
    else:
        t["function"]["parameters"]["properties"] = _shuffle_dict_keys(
            t["function"]["parameters"]["properties"], rng)
        where = "parameters.properties"
    return {"element": "tools", "tool_index": ti, "detail": f"key_order:{where}"}


def _ev_f3(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    present = {_tool_name(t) for t in st.tools}
    addable = [t for t in st.tool_pool if _tool_name(t) not in present]
    ops = []
    if len(st.tools) >= 2:
        ops += ["reorder", "remove"]
    if addable:
        ops.append("add")
    if not ops:
        return None
    op = rng.choice(ops)
    if op == "reorder":
        order = list(range(len(st.tools)))
        new = order[:]
        while new == order:
            rng.shuffle(new)
        st.tools = [st.tools[i] for i in new]
    elif op == "remove":
        st.tools.pop(rng.randrange(len(st.tools)))
    else:
        st.tools.insert(rng.randint(0, len(st.tools)), copy.deepcopy(rng.choice(addable)))
    return {"element": "tools", "detail": op}


def _ev_f4(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    s = st.system
    i = s.idx("profile")
    old = s.sections[i][1]
    new = old
    while new == old:
        new = _profile(rng)
    s.sections[i][1] = new
    return {"element": "system", "section": "profile", "detail": "user_name"}


def _ev_f5(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    s = st.system
    k = rng.randrange(3)
    i = s.idx(f"optional{k}")
    s.sections[i][2] = not s.sections[i][2]
    return {"element": "system", "section": f"optional{k}",
            "detail": "enabled" if s.sections[i][2] else "disabled"}


def _ev_f6(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    s = st.system
    op = rng.choice(("trailing_newline", "triple_newline", "double_space", "trailing_spaces"))
    if op == "trailing_newline":
        s.trailing += "\n"
        return {"element": "system", "detail": op}
    stable = [i for i, sec in enumerate(s.sections)
              if sec[0].startswith(("identity", "guideline")) and sec[2]]
    i = rng.choice(stable)
    if op == "triple_newline":
        if i == 0:
            return None
        s.joiners[i - 1] = s.joiners[i - 1] + "\n"
    elif op == "double_space":
        text = s.sections[i][1]
        pos = [p for p in range(len(text) - 1) if text[p] == "." and text[p + 1] == " "]
        if not pos:
            return None
        p = rng.choice(pos)
        s.sections[i][1] = text[: p + 1] + " " + text[p + 1:]
    else:
        s.sections[i][1] = s.sections[i][1] + "  "
    return {"element": "system", "section": s.sections[i][0], "detail": op}


def _ev_f7(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    s = st.system
    i = s.idx("rag")
    old = s.sections[i][1]
    new = old
    while new == old:
        new = _rag_block(rng)
    s.sections[i][1] = new
    return {"element": "system", "section": "rag", "detail": "retrieved_context_replaced"}


def _ev_f8(st: _SessionState, rng: random.Random, p_len: int) -> dict | None:
    # rewrite a turn that the reference request already contained
    cands = [i for i, m in enumerate(st.history[:p_len])
             if m["role"] in ("tool", "user") or (m["role"] == "assistant" and m.get("content"))]
    cands = [i for i in cands if len(st.history[i].get("content") or "") > 40
             and "[truncated]" not in st.history[i]["content"]]
    if not cands:
        return None
    i = rng.choice(cands)
    m = st.history[i]
    c = m["content"]
    m["content"] = c[: max(10, len(c) * 2 // 5)] + " ...[truncated]"
    return {"element": f"messages[{i + 1}]", "role": m["role"], "detail": "truncated"}


_EVENTS = {"F1": _ev_f1, "F2": _ev_f2, "F3": _ev_f3, "F4": _ev_f4,
           "F5": _ev_f5, "F6": _ev_f6, "F7": _ev_f7, "F8": _ev_f8}


def _fingerprint(st: _SessionState) -> str:
    msgs, tools = st.snapshot()
    return json.dumps([msgs, tools], ensure_ascii=False)


def _maybe_event(st: _SessionState, rng: random.Random, cfg: AgentLoadConfig,
                 scope: tuple[str, ...], p_len: int) -> tuple[str, dict] | None:
    u = rng.random()
    acc = 0.0
    chosen = None
    for f in scope:
        acc += cfg.faults[f]
        if u < acc:
            chosen = f
            break
    if chosen is None:
        return None
    before = _fingerprint(st)
    loc = _EVENTS[chosen](st, rng, p_len)
    if loc is None or _fingerprint(st) == before:
        return None  # event not applicable -> no byte change -> no label
    return chosen, loc


# --------------------------------------------------------------------------- #
# agent loop content
# --------------------------------------------------------------------------- #

def _user_msg(rng: random.Random) -> dict:
    return {"role": "user", "content": _paragraph(rng, rng.randint(60, 220))}


def _tool_step(rng: random.Random, tools: list[dict], call_seq: list[int]) -> list[dict]:
    if tools:
        t = rng.choice(tools)
        name = _tool_name(t)
        props = list(t["function"]["parameters"]["properties"])
        args = {p: (rng.randint(1, 100) if p.endswith("limit")
                    else f"{rng.choice(_NOUNS)}-{rng.randint(1, 999)}") for p in props}
    else:
        name, args = "noop", {"reason": "none", "level": 1}
    call_seq[0] += 1
    call_id = f"call_{call_seq[0]:04d}"
    asst = {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}
    if rng.random() < 0.5:
        result = json.dumps({"status": "ok", "items": [_sentence(rng, 3, 7) for _ in range(rng.randint(1, 4))]})
    else:
        result = _paragraph(rng, rng.randint(150, 600))
    return [asst, {"role": "tool", "tool_call_id": call_id, "content": result}]


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def generate(cfg: AgentLoadConfig) -> tuple[list[dict], list[dict]]:
    """Return (requests, labels). Requests are sorted by timestamp."""
    rng = random.Random(cfg.seed)
    n_pool = cfg.tools + 4
    names: list[str] = []
    while len(names) < n_pool:
        n = f"{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}"
        if n not in names:
            names.append(n)
    pool = [_make_tool(rng, n) for n in names]
    base_tools, extra_tools = pool[: cfg.tools], pool[cfg.tools:]

    raw: list[tuple[datetime, int, int, dict, dict | None]] = []
    prev_initial: _SessionState | None = None
    prev_first_id: str | None = None
    start = _BASE_TIME

    for si in range(cfg.sessions):
        sid = f"sess-{si:04d}"
        if si:
            # strictly increasing session starts -> "previous session" is well defined
            start = start + timedelta(seconds=cfg.session_gap_s * rng.uniform(0.5, 1.5) + 0.001)
        label_first: dict | None = None
        if prev_initial is None:
            st = _SessionState(_initial_system(cfg, rng), copy.deepcopy(base_tools),
                               copy.deepcopy(extra_tools))
        else:
            st = copy.deepcopy(prev_initial)
            st.history = []
            ev = _maybe_event(st, rng, cfg, CROSS_SESSION_FAULTS, 0)
            if ev:
                label_first = {"fault": ev[0], "scope": "cross_session",
                               "reference_request_id": prev_first_id, "location": ev[1]}
        if cfg.client_tool_order == "per_session_shuffle" and len(st.tools) > 1:
            # separate RNG: does not perturb the main stream (other fixtures stay byte-identical)
            srng = random.Random(f"{cfg.seed}:client_tool_order:{si}")
            srng.shuffle(st.tools)
        initial = copy.deepcopy(st)

        t = start
        step = 0
        call_seq = [0]
        pending: list[dict] = []   # messages produced by the previous step's response
        first_id = None

        def emit(label: dict | None) -> str:
            nonlocal step, t
            msgs, tools = st.snapshot()
            rid = f"{sid}-r{step:03d}"
            body: dict[str, Any] = {"request_id": rid, "session_id": sid, "timestamp": _iso(t),
                                    "model": cfg.model, "messages": msgs}
            if tools:
                body["tools"] = tools
            raw.append((t, si, step, body, label))
            step += 1
            t = t + timedelta(seconds=cfg.step_gap_s * rng.uniform(0.3, 1.7))
            if rng.random() < cfg.retry_rate:
                retry = copy.deepcopy(body)
                retry["request_id"] = f"{sid}-r{step:03d}"
                retry["timestamp"] = _iso(t)
                raw.append((t, si, step, retry, None))
                step += 1
                t = t + timedelta(seconds=cfg.step_gap_s * rng.uniform(0.3, 1.7))
            return rid

        for turn in range(cfg.turns):
            n_steps = rng.randint(cfg.tool_burst_min, cfg.tool_burst_max)
            for k in range(n_steps + 1):
                p_len = len(st.history)
                st.history.extend(pending)
                if turn == 0 and k == 0 and not pending:
                    st.history.append(_user_msg(rng))
                    rid = emit(label_first)
                    first_id = rid
                else:
                    if k == 0:
                        st.history.append(_user_msg(rng))
                    ev = _maybe_event(st, rng, cfg, WITHIN_SESSION_FAULTS, p_len)
                    lab = None
                    if ev:
                        lab = {"fault": ev[0], "scope": "within_session",
                               "reference_request_id": None, "location": ev[1]}
                    emit(lab)
                pending = []
                if k < n_steps:
                    pending = _tool_step(rng, st.tools, call_seq)
                else:
                    pending = [{"role": "assistant", "content": _paragraph(rng, rng.randint(80, 300))}]
        prev_initial = initial
        prev_first_id = first_id

    raw.sort(key=lambda r: (r[0], r[1], r[2]))
    requests: list[dict] = []
    labels: list[dict] = []
    prev_in_session: dict[str, str] = {}
    for idx, (_, _, _, body, lab) in enumerate(raw):
        requests.append(body)
        sid = body["session_id"]
        if lab is not None:
            lab = dict(lab)
            if lab["scope"] == "within_session":
                lab["reference_request_id"] = prev_in_session.get(sid)
            labels.append({"request_index": idx, "request_id": body["request_id"],
                           "session_id": sid, **lab})
        prev_in_session[sid] = body["request_id"]
    return requests, labels


def write_trace(requests: list[dict], labels: list[dict], out_path: str,
                labels_path: str | None = None, cfg: AgentLoadConfig | None = None) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if labels_path is None:
        labels_path = (out_path[:-6] if out_path.endswith(".jsonl") else out_path) + ".labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        if cfg is not None:
            f.write(json.dumps({"_meta": {"generator": "prefixanvil.agentload",
                                          "config": asdict(cfg)}}, ensure_ascii=False) + "\n")
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    return labels_path


def load_labels(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                continue
            out.append(d)
    return out
