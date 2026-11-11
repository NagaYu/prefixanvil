"""First-divergence detection and F1–F8 cause classification.

For every request C we pick a *reference* request P (see :func:`pair_requests`)
and ask: where does C's prompt first stop matching P's, and why?

Only the FIRST divergence matters for prefix caching (the hash chain is broken
from there on), so the analysis is local to the first differing element and
the first differing hunk inside it. Rules are ordered from most to least
specific; anything that does not match a rule is reported as ``unknown``.

Labels: ``none`` (no fragmentation: C extends P, or the divergence is the
natural start of a different conversation), ``F1``..``F8``, ``unknown``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Any

from .ingest import Request
from .render import Rendered, Renderer, RenderError
from .simulator import aligned

FAULT_NAMES = {
    "F1": "volatile value (time/date/UUID/sequential id)",
    "F2": "non-deterministic serialization (JSON-equal, bytes differ)",
    "F3": "tool-set churn (add/remove/reorder)",
    "F4": "per-user interpolation (short name-like span in system)",
    "F5": "conditional system section (present/absent)",
    "F6": "format drift (whitespace/newline only, template change)",
    "F7": "RAG insertion position (long block before stable part)",
    "F8": "history rewrite (past turn changed)",
    "unknown": "divergence found but no rule matched",
}
LABELS = tuple(f"F{i}" for i in range(1, 9))


@dataclass
class ClassifyConfig:
    block_size: int = 16
    rag_min_chars: int = 512          # a "long block" (≈128 tokens at 4 chars/token)
    max_span_chars: int = 40          # F4: max length of one differing span
    max_span_count: int = 6           # F4: max number of differing word spans
    context_chars: int = 32           # chars that must match again after a volatile/ws hunk
    nosession_window: int = 16        # candidates for best-match pairing without session_id


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def lcp(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    # galloping compare on chunks first (fast path for long equal prefixes)
    step = 256
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _content_text(m: dict) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, list):
        return "".join(p.get("text", "") if p.get("type") == "text" else f"<{p.get('type')}>" for p in c)
    return str(c)


def _msg_key(m: dict) -> str:
    return json.dumps([m.get("role"), _content_text(m), m.get("tool_calls"), m.get("name"),
                       m.get("tool_call_id")], ensure_ascii=False)


def _tool_name(t: Any) -> str | None:
    if isinstance(t, dict):
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        if isinstance(t.get("name"), str):
            return t["name"]
    return None


_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_WD = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_VOLATILE = [
    ("iso_datetime", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")),
    ("uuid", re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")),
    ("date", re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}")),
    ("weekday_date", re.compile(rf"\b{_WD}[a-z]*\.?,? (?:{_MON}[a-z]*\.? \d{{1,2}}|\d{{1,2}} {_MON}[a-z]*\.?)(?:,? \d{{4}})?")),
    ("month_date", re.compile(rf"\b{_MON}[a-z]*\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}\b")),
    ("day_month_year", re.compile(rf"\b\d{{1,2}} {_MON}[a-z]*\.? \d{{4}}\b")),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b")),
    ("epoch", re.compile(r"\b1\d{9}(?:\d{3})?\b")),
    ("hex_id", re.compile(r"\b[0-9a-fA-F]{16,}\b")),
    ("prefixed_id", re.compile(r"\b[A-Za-z]+[-_#]?\d{3,}\b")),
    ("number", re.compile(r"\b\d{3,}\b")),
]
_VOLATILE_BY_NAME = dict(_VOLATILE)

# Subset used where a false positive would *change* a prompt (L2 relocation):
# no bare numbers / clock times, and request-id style tokens need >= 5 digits.
VOLATILE_STRONG = [(n, _VOLATILE_BY_NAME[n]) for n in (
    "iso_datetime", "uuid", "date", "weekday_date", "month_date", "day_month_year", "epoch", "hex_id")] + [
    ("request_id", re.compile(r"\b[A-Za-z]+[-_#]?\d{5,}\b")),
]


def _match_covering(rx: re.Pattern, s: str, k: int) -> re.Match | None:
    lo = max(0, k - 80)
    for m in rx.finditer(s, lo, min(len(s), k + 80)):
        if m.start() <= k < m.end():
            return m
        if m.start() > k:
            break
    return None


def _check_json(a: str, b: str, k: int) -> str | None:
    dec = json.JSONDecoder()
    tried = 0
    pos = k if k < len(a) else len(a) - 1
    while pos >= 0 and tried < 64:
        if a[pos] in "{[":
            tried += 1
            try:
                oa, ea = dec.raw_decode(a, pos)
                ob, eb = dec.raw_decode(b, pos)
            except (json.JSONDecodeError, ValueError):
                pos -= 1
                continue
            if ea > k or eb > k:          # this value covers the divergence
                if oa == ob and a[pos:ea] != b[pos:eb]:
                    return "json_equal_bytes_differ"
                return None               # covering value genuinely differs
        pos -= 1
    return None


def _check_whitespace(a: str, b: str, k: int, ctx: int) -> str | None:
    at = a[k] if k < len(a) else ""
    bt = b[k] if k < len(b) else ""
    if not ((at and at.isspace()) or (bt and bt.isspace())):
        return None
    m = len(re.sub(r"\s+", "", a[:k]))
    na = re.sub(r"\s+", "", a)
    nb = re.sub(r"\s+", "", b)
    need = min(ctx, len(na) - m, len(nb) - m)
    if lcp(na, nb) >= m + max(0, need):
        return "whitespace_only"
    return None


def _check_volatile(a: str, b: str, k: int, ctx: int) -> str | None:
    for name, rx in _VOLATILE:
        ma = _match_covering(rx, a, k)
        mb = _match_covering(rx, b, k)
        if not ma or not mb:
            continue
        ra, rb = a[ma.end():], b[mb.end():]
        n = min(ctx, len(ra), len(rb))
        if ra[:n] == rb[:n] and (n > 0 or (not ra and not rb)):
            return name
    return None


_WORD_RX = re.compile(r"\w+|[^\w\s]|\s+")
_NAMEISH = re.compile(r"[\w .'@-]*[^\W\d_][\w .'@-]*")


def _check_short_spans(a: str, b: str, cfg: ClassifyConfig) -> bool:
    wa, wb = _WORD_RX.findall(a), _WORD_RX.findall(b)
    ops = [op for op in SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops or len(ops) > cfg.max_span_count:
        return False
    for _, i1, i2, j1, j2 in ops:
        for span in ("".join(wa[i1:i2]), "".join(wb[j1:j2])):
            if len(span) > cfg.max_span_chars or "\n" in span:
                return False
            if span.strip() and not _NAMEISH.fullmatch(span.strip()):
                return False
    return True


def _line_insert_delete(a: str, b: str) -> str | None:
    """Sections without blank-line separators: whole lines added/removed only?"""
    la, lb = a.split("\n"), b.split("\n")
    ops = [op for op in SequenceMatcher(None, la, lb, autojunk=False).get_opcodes() if op[0] != "equal"]
    kinds = {op[0] for op in ops}
    # all-insert or all-delete; a delete+insert pair is a reorder/rewrite, not a toggle
    if kinds == {"insert"}:
        return "lines_added"
    if kinds == {"delete"}:
        return "lines_removed"
    return None


def _split_sections(s: str) -> list[str]:
    return re.split(r"\n{2,}", s)


def analyze_text(a: str, b: str, cfg: ClassifyConfig, system: bool) -> tuple[str, str] | None:
    """Classify the first differing hunk between two strings.

    Returns (label, detail) or None if no rule matched (callers decide the
    fallback: ``unknown`` for system text, ``F8`` for past turns).
    """
    k = lcp(a, b)
    if k == len(a) and k == len(b):
        return None
    d = _check_json(a, b, k)
    if d:
        return "F2", d
    d = _check_whitespace(a, b, k, cfg.context_chars)
    if d:
        return "F6", d
    d = _check_volatile(a, b, k, cfg.context_chars)
    if d:
        return "F1", f"volatile:{d}"
    if not system:
        return None

    sa, sb = _split_sections(a), _split_sections(b)
    ops = SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            continue
        stable_after = any(t == "equal" for t, *_ in ops[n + 1:])
        la = sum(len(x) for x in sa[i1:i2])
        lb = sum(len(x) for x in sb[j1:j2])
        if tag in ("insert", "delete"):
            what = "added" if tag == "insert" else "removed"
            if max(la, lb) >= cfg.rag_min_chars and stable_after:
                return "F7", f"long_block_{what}_before_stable_part"
            return "F5", f"section_{what}"
        # replace
        if max(la, lb) >= cfg.rag_min_chars and stable_after:
            return "F7", "long_block_changed_before_stable_part"
        ja, jb = "\n\n".join(sa[i1:i2]), "\n\n".join(sb[j1:j2])
        line_level = _line_insert_delete(ja, jb)
        if line_level:
            return "F5", line_level
        if _check_short_spans(ja, jb, cfg):
            return "F4", "short_name_like_spans"
        if (i2 - i1) != (j2 - j1):
            return "F5", "section_structure_changed"
        return "unknown", "unclassified_system_change"
    return "unknown", "unclassified_system_change"


def analyze_tools(pt: list[dict], ct: list[dict], cfg: ClassifyConfig) -> tuple[str, str]:
    pn = [_tool_name(t) for t in pt]
    cn = [_tool_name(t) for t in ct]
    if Counter(pn) != Counter(cn):
        added = sorted(str(x) for x in (Counter(cn) - Counter(pn)))
        removed = sorted(str(x) for x in (Counter(pn) - Counter(cn)))
        return "F3", f"tools_added={added} removed={removed}"
    if pn != cn:
        return "F3", "tools_order_only"
    for x, y in zip(pt, ct):
        sx, sy = json.dumps(x, ensure_ascii=False), json.dumps(y, ensure_ascii=False)
        if sx == sy:
            continue
        if x == y:
            return "F2", "tool_schema_key_order"
        r = analyze_text(sx, sy, cfg, system=False)
        if r:
            return r
        return "unknown", "tool_definition_changed"
    return "unknown", "tools_differ_unlocated"


# --------------------------------------------------------------------------- #
# per-request analysis items and pairing
# --------------------------------------------------------------------------- #

@dataclass
class Item:
    req: Request
    rendered: Rendered
    _nogen_text: str | None = None
    _nogen_len: int | None = None
    _static_len: int | None = None


@dataclass
class PairResult:
    request_index: int
    request_id: str | None
    reference_index: int | None
    scope: str                       # within_session | cross_session | nosession
    label: str                       # none | F1..F8 | unknown
    detail: str
    element: str | None              # tools | messages[i] | template | None
    first_divergence_token: int | None
    first_divergence_char: int | None
    prompt_tokens: int
    expected_shared_tokens: int      # block-aligned reuse an ideal (stable) prefix would allow vs. reference
    actual_shared_tokens: int        # block-aligned common prefix with the reference
    lost_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


class Classifier:
    def __init__(self, renderer: Renderer, cfg: ClassifyConfig | None = None):
        self.r = renderer
        self.cfg = cfg or ClassifyConfig()

    # -- lazily computed renders -------------------------------------------
    def _nogen(self, it: Item) -> str:
        if it._nogen_text is None:
            it._nogen_text = self.r.text(it.req.messages, it.req.tools, add_generation_prompt=False)
        return it._nogen_text

    def _nogen_tokens(self, it: Item) -> int:
        if it._nogen_len is None:
            it._nogen_len = len(self.r.tok.encode(self._nogen(it)))
        return it._nogen_len

    def _static_tokens(self, it: Item) -> int:
        """Tokens of the static prefix: leading system messages + tools."""
        if it._static_len is None:
            msgs = it.req.messages
            s = 0
            while s < len(msgs) and msgs[s].get("role") in ("system", "developer"):
                s += 1
            if s == 0 and not it.req.tools:
                it._static_len = 0
            else:
                try:
                    txt = self.r.text(msgs[:s], it.req.tools, add_generation_prompt=False)
                    it._static_len = len(self.r.tok.encode(txt))
                except RenderError:
                    it._static_len = 0
        return it._static_len

    # -- main ------------------------------------------------------------------
    def classify_pair(self, P: Item, C: Item, scope: str) -> PairResult:
        B = self.cfg.block_size
        ptok, ctok = P.rendered.token_ids, C.rendered.token_ids
        n = len(ctok)
        tok_lcp = lcp(ptok, ctok)
        actual = aligned(min(tok_lcp, max(0, n - 1)), B)

        label, detail, element, d_char = self._cause(P, C, scope)

        if scope == "cross_session":
            exp_raw = min(self._static_tokens(P), self._static_tokens(C))
        else:
            exp_raw = self._nogen_tokens(P)
        expected = aligned(min(exp_raw, max(0, n - 1)), B)
        if label == "none":
            expected = min(expected, actual)
        expected = max(expected, actual)
        div_tok = tok_lcp if tok_lcp < min(len(ptok), n) else None
        return PairResult(C.req.index, C.req.request_id, P.req.index, scope, label, detail, element,
                          div_tok, d_char, n, expected, actual, expected - actual)

    def _cause(self, P: Item, C: Item, scope: str) -> tuple[str, str, str | None, int | None]:
        cfg = self.cfg
        if P.rendered.template_version != C.rendered.template_version:
            return "F6", "template_version_changed", "template", 0
        pn = self._nogen(P)
        d_char = lcp(pn, C.rendered.text)
        if d_char >= len(pn):
            return "none", "extends_reference", None, None

        pm, cm = P.req.messages, C.req.messages
        tools_equal = json.dumps(P.req.tools or [], ensure_ascii=False) == \
            json.dumps(C.req.tools or [], ensure_ascii=False)
        j = None
        for i in range(min(len(pm), len(cm))):
            if _msg_key(pm[i]) != _msg_key(cm[i]):
                j = i
                break
        if j is None and len(cm) < len(pm):
            j = len(cm)

        tools_first = False
        if not tools_equal:
            if j is None:
                tools_first = True
            else:
                try:
                    cf = self.r.text(cm, P.req.tools, add_generation_prompt=True)
                    tools_first = lcp(pn, cf) > d_char
                except RenderError:
                    tools_first = False
        if tools_first:
            lab, det = analyze_tools(P.req.tools or [], C.req.tools or [], cfg)
            return lab, det, "tools", d_char
        if j is None:
            # structurally identical within P's range, yet the rendering differs
            # e.g. a template that renders past turns differently from the turn it
            # just generated (reasoning stripping, last-turn special cases)
            return "F6", "template_rerenders_history", "template", d_char

        el = f"messages[{j}]"
        if j >= len(cm):
            return ("none", "different_conversation", el, d_char) if scope == "cross_session" \
                else ("F8", "message_deleted", el, d_char)
        a, b = pm[j], cm[j]
        if a.get("role") != b.get("role"):
            return ("none", "different_conversation", el, d_char) if scope == "cross_session" \
                else ("F8", "role_sequence_changed", el, d_char)
        is_sys = a.get("role") in ("system", "developer")
        if not is_sys and scope == "cross_session":
            return "none", "different_conversation", el, d_char
        ta, tb = _content_text(a), _content_text(b)
        if ta != tb:
            r = analyze_text(ta, tb, cfg, system=is_sys)
            if r:
                return r[0], r[1], el, d_char
            return ("unknown", "unclassified_system_change", el, d_char) if is_sys \
                else ("F8", "past_turn_content_changed", el, d_char)
        # content equal: tool_calls / metadata differ
        tca, tcb = a.get("tool_calls") or [], b.get("tool_calls") or []
        for x, y in zip(tca, tcb):
            if json.dumps(x, ensure_ascii=False) == json.dumps(y, ensure_ascii=False):
                continue
            fx, fy = x.get("function", {}), y.get("function", {})
            if fx.get("name") != fy.get("name"):
                return "F8", "past_tool_call_changed", el, d_char
            ax, ay = fx.get("arguments"), fy.get("arguments")
            try:
                ox = json.loads(ax) if isinstance(ax, str) else ax
                oy = json.loads(ay) if isinstance(ay, str) else ay
                if ox == oy:
                    return "F2", "tool_call_arguments_reserialized", el, d_char
            except (json.JSONDecodeError, TypeError):
                pass
            if isinstance(ax, str) and isinstance(ay, str):
                r = analyze_text(ax, ay, cfg, system=False)
                if r:
                    return r[0], r[1], el, d_char
            return "F8", "past_tool_call_changed", el, d_char
        if len(tca) != len(tcb):
            return "F8", "past_tool_calls_changed", el, d_char
        return "F8", "past_message_metadata_changed", el, d_char


def pair_requests(items: list[Item], cfg: ClassifyConfig) -> list[tuple[int, int, str]]:
    """Choose a reference for each request. Returns (ref_pos, cur_pos, scope).

    * within a session: the immediately preceding request of the same session;
    * first request of a session: the previous session's first request with the
      same model (cross-session sharing of the static prefix);
    * no session_id: the most similar (longest common rendered prefix) of the
      last ``nosession_window`` session-less requests of the same model.
    """
    pairs = []
    last_in_session: dict[str, int] = {}
    last_session_start: dict[str, int] = {}
    nosession: dict[str, list[int]] = defaultdict(list)
    for pos, it in enumerate(items):
        r = it.req
        if r.session_id:
            if r.session_id in last_in_session:
                pairs.append((last_in_session[r.session_id], pos, "within_session"))
            else:
                if r.model in last_session_start:
                    pairs.append((last_session_start[r.model], pos, "cross_session"))
                last_session_start[r.model] = pos
            last_in_session[r.session_id] = pos
        else:
            cands = nosession[r.model][-cfg.nosession_window:]
            if cands:
                best = max(cands, key=lambda q: (lcp(items[q].rendered.token_ids, it.rendered.token_ids), q))
                P = items[best]
                # same conversation if the first non-system message matches
                same = any(_msg_key(a) == _msg_key(b) and a.get("role") not in ("system", "developer")
                           for a, b in zip(P.req.messages, r.messages))
                pairs.append((best, pos, "within_session" if same else "cross_session"))
            nosession[r.model].append(pos)
    return pairs


def classify_all(items: list[Item], classifier: Classifier) -> list[PairResult]:
    out = []
    for ref, cur, scope in pair_requests(items, classifier.cfg):
        try:
            out.append(classifier.classify_pair(items[ref], items[cur], scope))
        except RenderError as e:
            C = items[cur]
            out.append(PairResult(C.req.index, C.req.request_id, items[ref].req.index, scope,
                                  "unknown", f"render_error:{e}", None, None, None,
                                  len(C.rendered.token_ids), 0, 0, 0))
    return out


# --------------------------------------------------------------------------- #
# evaluation against AgentLoad ground truth
# --------------------------------------------------------------------------- #

def evaluate(results: list[PairResult], labels: list[dict]) -> dict:
    by_id = {lab["request_id"]: lab["fault"] for lab in labels if lab.get("request_id")}
    by_idx = {lab["request_index"]: lab["fault"] for lab in labels if "request_index" in lab}
    conf: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        truth = by_id.get(r.request_id) if r.request_id is not None and by_id else by_idx.get(r.request_index)
        truth = truth or "none"
        conf[truth][r.label] += 1
    per = {}
    for f in LABELS:
        tp = conf[f][f]
        support = sum(conf[f].values())
        predicted = sum(c[f] for c in conf.values())
        per[f] = {
            "support": support, "predicted": predicted, "tp": tp,
            "recall": tp / support if support else None,
            "precision": tp / predicted if predicted else None,
        }
    labelled = {lab["request_id"] for lab in labels if lab.get("request_id")}
    evaluated = {r.request_id for r in results}
    return {
        "per_fault": per,
        "confusion": {t: dict(c) for t, c in sorted(conf.items())},
        "labels_without_pair": len(labelled - evaluated),
    }
