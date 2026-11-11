"""Stateless, deterministic core API with versioned JSON I/O.

This is the seam every caller (proxy, equivalence gate, a future SaaS) goes
through. The core keeps NO state between calls: per-session state is a
:class:`~prefixanvil.canonical.SessionContext` owned by the caller, passed in
as JSON and returned (updated) as JSON.

``transform`` (schema ``prefixanvil.transform.request/v1`` ->
``prefixanvil.transform.result/v1``)::

    {"schema": "prefixanvil.transform.request/v1",
     "mode": "L0" | "L1" | "L2",
     "request": {<OpenAI chat.completions body>},
     "session_context": {<prefixanvil.session-context/v1>} | null}

    -> {"schema": "prefixanvil.transform.result/v1", "mode": ...,
        "request": {<body to forward>}, "changed": bool,
        "applied": [...], "skipped": [...], "notes": [...],
        "would_apply": [...],                # L0 only: what L1 would do
        "session_context": {...} | null,     # pass with the next request of the session
        "before_sha256": "...", "after_sha256": "..."}

Fail-open: any invalid input or internal error returns the original request
unchanged with a ``skipped`` entry ``{"rule": "*", "reason": ...}``.
Transform logs carry rule ids, JSON paths and SHA-256 prefixes only.

``Observer.observe`` runs the analyzer's first-divergence classifier on a
(reference, request) pair (``prefixanvil.observe.request/v1`` ->
``prefixanvil.observe.result/v1``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .canonical import SessionContext, TransformLog, canonicalize, next_session_context
from .ingest import Request, ValidationError, validate_obj
from .relocate import relocate_volatile_lines

TRANSFORM_REQUEST_SCHEMA = "prefixanvil.transform.request/v1"
TRANSFORM_RESULT_SCHEMA = "prefixanvil.transform.result/v1"
OBSERVE_REQUEST_SCHEMA = "prefixanvil.observe.request/v1"
OBSERVE_RESULT_SCHEMA = "prefixanvil.observe.result/v1"
MODES = ("L0", "L1", "L2")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _to_request(body: dict) -> Request:
    model, messages, tools, _, _, _ = validate_obj(body)
    return Request(0, 0, model, messages, tools, None, None, None)


def _write_back(body: dict, req: Request) -> dict:
    out = copy.deepcopy(body)
    out["messages"] = req.messages
    if req.tools is not None:
        out["tools"] = req.tools
    return out


def _fail_open(payload: Any, reason: str) -> dict:
    body = payload.get("request") if isinstance(payload, dict) else None
    ctx = payload.get("session_context") if isinstance(payload, dict) else None
    mode = payload.get("mode") if isinstance(payload, dict) else None
    h = sha256_json(body) if body is not None else None
    return {"schema": TRANSFORM_RESULT_SCHEMA, "mode": mode, "request": body, "changed": False,
            "applied": [], "skipped": [{"rule": "*", "path": "request", "reason": reason}],
            "notes": [], "would_apply": [], "session_context": ctx,
            "before_sha256": h, "after_sha256": h}


def transform(payload: dict, *, _canonicalize=canonicalize) -> dict:
    """Pure function: transform one request. ``_canonicalize`` is a test seam."""
    try:
        if not isinstance(payload, dict) or payload.get("schema") != TRANSFORM_REQUEST_SCHEMA:
            return _fail_open(payload, "unsupported_schema")
        mode = payload.get("mode")
        if mode not in MODES:
            return _fail_open(payload, "unsupported_mode")
        body = payload.get("request")
        try:
            req = _to_request(body)
        except ValidationError as e:
            return _fail_open(payload, f"invalid_request:{e.reason}")
        ctx = SessionContext.from_json(payload.get("session_context")) \
            if payload.get("session_context") is not None else None

        log = TransformLog()
        work = req
        if mode == "L2":
            work = relocate_volatile_lines(work, log)
        canon, l1log = _canonicalize(work, ctx)
        log.applied += l1log.applied
        log.skipped += l1log.skipped
        log.notes += l1log.notes
        next_ctx = next_session_context(canon).to_json() if ctx is not None else None
        before = sha256_json(body)
        if mode == "L0":
            return {"schema": TRANSFORM_RESULT_SCHEMA, "mode": mode, "request": body, "changed": False,
                    "applied": [], "skipped": [], "notes": log.notes, "would_apply": log.applied,
                    "would_skip": log.skipped, "session_context": next_ctx,
                    "before_sha256": before, "after_sha256": before}
        out = _write_back(body, canon) if log.applied else body
        after = sha256_json(out)
        return {"schema": TRANSFORM_RESULT_SCHEMA, "mode": mode, "request": out,
                "changed": after != before, "applied": log.applied, "skipped": log.skipped,
                "notes": log.notes, "would_apply": [], "session_context": next_ctx,
                "before_sha256": before, "after_sha256": after}
    except Exception as e:  # fail-open on anything unexpected
        return _fail_open(payload, f"internal_error:{type(e).__name__}")


def sanitize_detail(detail: str) -> str:
    """Classifier details may embed tool names; keep only the code part."""
    if detail.startswith("render_error"):
        return "render_error"
    return detail.split("=", 1)[0]


class Observer:
    """Classification on the fly. Holds only configuration (the renderer)."""

    def __init__(self, renderer, cfg=None):
        from .classify import ClassifyConfig, Classifier
        self.renderer = renderer
        self.classifier = Classifier(renderer, cfg or ClassifyConfig())

    def observe(self, payload: dict) -> dict:
        from .classify import Item
        try:
            if payload.get("schema") != OBSERVE_REQUEST_SCHEMA:
                raise ValueError("unsupported_schema")
            cur = _to_request(payload["request"])
            ref_body = payload.get("reference")
            if ref_body is None:
                return {"schema": OBSERVE_RESULT_SCHEMA, "label": None, "detail": "no_reference"}
            ref = _to_request(ref_body)
            P = Item(ref, self.renderer.render(ref))
            C = Item(cur, self.renderer.render(cur))
            r = self.classifier.classify_pair(P, C, payload.get("scope", "within_session"))
            return {"schema": OBSERVE_RESULT_SCHEMA, "label": r.label, "detail": sanitize_detail(r.detail),
                    "element": r.element, "scope": r.scope,
                    "first_divergence_token": r.first_divergence_token,
                    "prompt_tokens": r.prompt_tokens, "lost_tokens": r.lost_tokens}
        except Exception as e:
            return {"schema": OBSERVE_RESULT_SCHEMA, "label": None,
                    "detail": f"observe_error:{type(e).__name__}"}
