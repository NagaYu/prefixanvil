"""L1 canonicalization: deterministic, meaning-preserving rewrites only.

Rules (IDs are stable and appear in the transform log):

* ``L1.tools.sort_keys``      recursive key sort of every tool definition
                              (JSON objects are unordered; arrays untouched).
* ``L1.tools.sort_by_name``   stateless sort of the tools array by function name
                              (only without a session context = Phase-1 analyzer
                              what-if mode; a note records it). Requires every
                              tool to have a unique name (else skipped, fail-open).
* ``L1.tools.session_stable_order`` (Phase 2, with a :class:`SessionContext`):
                              the session's first request keeps the client order;
                              later requests with an unchanged tool-name multiset
                              reuse the order emitted for the previous request
                              (skip ``session_order_pinned`` for the name sort);
                              when the multiset changed the name sort is skipped
                              with ``tool_set_changed`` and retained tools keep the
                              previously emitted order, new tools are appended.
* ``L1.json_string.normalize`` re-serialize strings that hold JSON
                              (``tool_calls[].function.arguments`` and
                              tool-role ``content`` that parses as an object or
                              array) with sorted keys and fixed separators.
* ``L1.system.whitespace``    in system/developer string content, outside code:
                              strip trailing whitespace per line, collapse runs of
                              spaces/tabs after the first non-blank char, collapse
                              3+ newlines into 2, strip trailing newlines. Lines in
                              ``` / ~~~ fences are untouched; lines indented by
                              >= 4 spaces / a tab keep their inner spacing (Phase 2,
                              found by the equivalence gate).

Anything with an unexpected shape is passed through unchanged and a
``skipped`` entry with the reason is logged (fail-open). Input objects are
never mutated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .ingest import Request

RULES = ("L1.tools.sort_keys", "L1.tools.sort_by_name", "L1.tools.session_stable_order",
         "L1.json_string.normalize", "L1.system.whitespace")


def _h(obj: Any) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


SESSION_CONTEXT_SCHEMA = "prefixanvil.session-context/v1"


@dataclass(frozen=True)
class SessionContext:
    """Caller-owned per-session state passed explicitly into the core.

    ``prev_tool_names``: sorted tool-name multiset of the previous request in
    the session (``None`` = this is the session's first request). Unnamed tools
    count as ``""``.
    """
    prev_tool_names: tuple[str, ...] | None = None
    prev_tool_order: tuple[str, ...] | None = None   # order L1 emitted for the previous request

    def to_json(self) -> dict:
        return {"schema": SESSION_CONTEXT_SCHEMA,
                "prev_tool_names": None if self.prev_tool_names is None else list(self.prev_tool_names),
                "prev_tool_order": None if self.prev_tool_order is None else list(self.prev_tool_order)}

    @classmethod
    def from_json(cls, d: dict | None) -> "SessionContext":
        if not d:
            return cls()
        if d.get("schema") != SESSION_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported session context schema {d.get('schema')!r}")
        names, order = d.get("prev_tool_names"), d.get("prev_tool_order")
        return cls(None if names is None else tuple(str(x) for x in names),
                   None if order is None else tuple(str(x) for x in order))


def tool_name_multiset(tools: list[dict] | None) -> tuple[str, ...]:
    return tuple(sorted((_tool_name(t) or "") for t in (tools or [])))


def next_session_context(emitted: Request) -> SessionContext:
    """Context for the *next* request of the same session (pure).

    Pass the request as it was emitted (after canonicalization) so that the
    next request can reuse the same tool order.
    """
    names = [_tool_name(t) for t in (emitted.tools or [])]
    order = None if any(n is None for n in names) or len(set(names)) != len(names) else tuple(names)
    return SessionContext(tool_name_multiset(emitted.tools), order)


@dataclass
class TransformLog:
    applied: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, rule: str, path: str, before: Any, after: Any) -> None:
        self.applied.append({"rule": rule, "path": path,
                             "before_sha256": _h(before), "after_sha256": _h(after)})

    def skip(self, rule: str, path: str, reason: str) -> None:
        self.skipped.append({"rule": rule, "path": path, "reason": reason})


def sort_keys_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sort_keys_recursive(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [sort_keys_recursive(x) for x in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))


_INNER_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
_FENCE = re.compile(r"^\s*(```|~~~)")
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)")


def normalize_system_whitespace(s: str) -> str:
    """Whitespace normalization that never touches code.

    Lines inside ``` / ~~~ fences (an unclosed fence protects the rest of the
    text) are kept byte-identical, as are blank lines inside fences. Lines
    indented by >= 4 spaces or a tab (Markdown code) keep their inner spacing.
    Elsewhere: trailing whitespace stripped, inner runs of spaces/tabs collapsed,
    3+ newlines collapsed to 2. Trailing newlines are stripped unless the text
    ends inside an open fence.
    """
    out: list[str] = []
    in_fence = prev_blank = False
    for line in s.split("\n"):
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
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue
        prev_blank = False
        out.append(line if _INDENTED_CODE.match(line) else _INNER_SPACES.sub(" ", line))
    res = "\n".join(out)
    return res if in_fence else res.rstrip("\n")


def _tool_name(t: Any) -> str | None:
    if not isinstance(t, dict):
        return None
    fn = t.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    if isinstance(t.get("name"), str):
        return t["name"]
    return None


def _normalize_json_string(s: Any) -> tuple[str | None, str | None]:
    """Return (normalized, skip_reason)."""
    if not isinstance(s, str):
        return None, "not_a_string"
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None, "not_json"
    if not isinstance(v, (dict, list)):
        return None, "json_scalar"
    return canonical_json(v), None


def canonicalize_tools(tools: list[dict] | None, log: TransformLog,
                       ctx: SessionContext | None = None) -> list[dict] | None:
    if tools is None:
        return None
    out = []
    for i, t in enumerate(tools):
        st = sort_keys_recursive(t)
        if json.dumps(st) != json.dumps(t):
            log.add("L1.tools.sort_keys", f"tools[{i}]", t, st)
        out.append(st)
    names = [_tool_name(t) for t in out]
    if any(n is None for n in names):
        log.skip("L1.tools.sort_by_name", "tools", "tool_without_name")
    elif len(set(names)) != len(names):
        log.skip("L1.tools.sort_by_name", "tools", "duplicate_tool_names")
    elif ctx is None:
        # Phase-1 behaviour (analyzer what-if without sessions): stateless name sort
        log.notes.append("tool_sort:stateless_no_session_context")
        out = _apply_order(out, sorted(names), "L1.tools.sort_by_name", log)
    elif ctx.prev_tool_names is None:
        # first request of a session: keep the client's order (sorting here moves
        # later add/remove divergences; see README "Session-aware tool order")
        log.notes.append("tool_sort:session_first_request_client_order_kept")
    elif ctx.prev_tool_names == tool_name_multiset(out):
        log.skip("L1.tools.sort_by_name", "tools", "session_order_pinned")
        if ctx.prev_tool_order is not None and sorted(ctx.prev_tool_order) == sorted(names):
            out = _apply_order(out, list(ctx.prev_tool_order), "L1.tools.session_stable_order", log)
    else:
        log.skip("L1.tools.sort_by_name", "tools", "tool_set_changed")
        if ctx.prev_tool_order is not None:
            # retained tools keep the previously emitted order, new ones are appended
            keep = set(names)
            kept = [n for n in ctx.prev_tool_order if n in keep]
            target = kept + [n for n in names if n not in set(kept)]
            out = _apply_order(out, target, "L1.tools.session_stable_order", log)
    return out


def _apply_order(tools: list[dict], target: list[str], rule: str, log: TransformLog) -> list[dict]:
    names = [_tool_name(t) for t in tools]
    if target == names:
        return tools
    by = dict(zip(names, tools))
    log.add(rule, "tools", names, target)
    return [by[n] for n in target]


def canonicalize_messages(messages: list[dict], log: TransformLog) -> list[dict]:
    out = []
    for i, m in enumerate(messages):
        m = copy.deepcopy(m)
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str):
                new = normalize_system_whitespace(content)
                if new != content:
                    log.add("L1.system.whitespace", f"messages[{i}].content", content, new)
                    m["content"] = new
            elif content is not None:
                log.skip("L1.system.whitespace", f"messages[{i}].content", "non_string_content")
        if role == "tool" and isinstance(content, str):
            new, reason = _normalize_json_string(content)
            if new is not None and new != content:
                log.add("L1.json_string.normalize", f"messages[{i}].content", content, new)
                m["content"] = new
        for j, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict) or "arguments" not in fn:
                log.skip("L1.json_string.normalize", f"messages[{i}].tool_calls[{j}]", "unexpected_shape")
                continue
            new, reason = _normalize_json_string(fn["arguments"])
            path = f"messages[{i}].tool_calls[{j}].function.arguments"
            if new is None:
                if reason != "not_a_string":   # dict arguments are rendered by tojson already
                    log.skip("L1.json_string.normalize", path, reason)
                elif isinstance(fn["arguments"], dict):
                    srt = sort_keys_recursive(fn["arguments"])
                    if json.dumps(srt) != json.dumps(fn["arguments"]):
                        log.add("L1.json_string.normalize", path, fn["arguments"], srt)
                        fn["arguments"] = srt
                continue
            if new != fn["arguments"]:
                log.add("L1.json_string.normalize", path, fn["arguments"], new)
                fn["arguments"] = new
        out.append(m)
    return out


def canonicalize(req: Request, ctx: SessionContext | None = None) -> tuple[Request, TransformLog]:
    """Apply L1. ``ctx`` is the caller-owned session context (None = stateless)."""
    log = TransformLog()
    try:
        tools = canonicalize_tools(copy.deepcopy(req.tools), log, ctx)
        messages = canonicalize_messages(req.messages, log)
    except Exception as e:  # fail-open on anything unexpected
        log = TransformLog()
        log.skip("*", "request", f"exception:{type(e).__name__}")
        return req, log
    new = Request(req.index, req.line_no, req.model, messages, tools,
                  req.session_id, req.timestamp, req.request_id)
    return new, log
