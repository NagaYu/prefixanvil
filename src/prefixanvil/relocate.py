"""L2 (Phase 2): exactly one structural transform.

``L2.system.relocate_volatile_lines``: lines of a system/developer message that
are *volatile-only* — a short label followed only by volatile values such as a
timestamp, date, UUID, epoch, hex id or request id (``Session clock: 2026-...``,
``Request id: 550e8400-...``) — are moved, in their original order, to the end
of that message. Nothing else is changed. Lines inside fenced code blocks are
never moved. Detection reuses the F1 detector's patterns, restricted to the
strong subset (:data:`prefixanvil.classify.VOLATILE_STRONG`): bare numbers and
clock times are not treated as volatile here, because a false positive would
reorder ordinary instructions.

This changes what the model sees (line order), so it is NOT meaning-preserving
by construction: the equivalence gate reports it as an allowed diff against the
mock upstream, and real-model acceptability is a Phase 3 measurement.
"""

from __future__ import annotations

import copy
import re

from .canonical import TransformLog
from .classify import VOLATILE_STRONG
from .ingest import Request

RULE = "L2.system.relocate_volatile_lines"
_LABEL_ONLY = re.compile(r"\s*(?:[-*]\s+)?(?:[^\W\d_][\w .'/()-]{0,40}?\s*[:=])?[\s,;|]*")
_FENCE = re.compile(r"^\s*(```|~~~)")


def is_volatile_only_line(line: str) -> bool:
    if not line.strip() or len(line) > 200:
        return False
    spans = []
    for _, rx in VOLATILE_STRONG:
        spans += [m.span() for m in rx.finditer(line)]
    if not spans:
        return False
    keep = [True] * len(line)
    for a, b in spans:
        for i in range(a, b):
            keep[i] = False
    rest = "".join(ch for ch, k in zip(line, keep) if k)
    return _LABEL_ONLY.fullmatch(rest) is not None


def relocate_text(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    stay, moved = [], []
    in_fence = False
    for ln in lines:
        if _FENCE.match(ln):
            in_fence = not in_fence
            stay.append(ln)
            continue
        if not in_fence and is_volatile_only_line(ln):
            moved.append(ln)
        else:
            stay.append(ln)
    if not moved:
        return text, 0
    body = "\n".join(stay).rstrip("\n")
    return (body + "\n\n" + "\n".join(moved)) if body else "\n".join(moved), len(moved)


def relocate_volatile_lines(req: Request, log: TransformLog) -> Request:
    msgs = []
    changed = False
    for i, m in enumerate(req.messages):
        if m.get("role") in ("system", "developer"):
            c = m.get("content")
            if isinstance(c, str):
                new, n = relocate_text(c)
                if n:
                    log.add(RULE, f"messages[{i}].content", c, new)
                    log.applied[-1]["lines_moved"] = n
                    m = dict(m, content=new)
                    changed = True
            elif c is not None:
                log.skip(RULE, f"messages[{i}].content", "non_string_content")
        msgs.append(m)
    if not changed:
        return req
    return Request(req.index, req.line_no, req.model, copy.deepcopy(msgs), req.tools,
                   req.session_id, req.timestamp, req.request_id)
