"""Load and validate OpenAI-compatible request logs (JSONL).

Each non-blank line must be a JSON object with at least ``model`` and
``messages``. ``tools``, ``session_id``, ``timestamp`` and ``request_id`` are
optional. Lines that fail validation are never dropped silently: they are
counted per reason (with sample line numbers) in :class:`IngestResult`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

VALID_ROLES = ("system", "developer", "user", "assistant", "tool", "function")
MAX_SAMPLES_PER_REASON = 5


@dataclass
class Request:
    index: int                    # 0-based index among *accepted* requests (replay order)
    line_no: int                  # 1-based line number in the source file
    model: str
    messages: list[dict]
    tools: list[dict] | None
    session_id: str | None
    timestamp: float | None
    request_id: str | None

    @property
    def group_key(self) -> str:
        return f"session:{self.session_id}" if self.session_id else f"nosession:{self.model}"


@dataclass
class IngestResult:
    requests: list[Request] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    rejected: Counter = field(default_factory=Counter)
    rejected_samples: dict[str, list[int]] = field(default_factory=dict)
    order: str = "file"

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    def reject(self, reason: str, line_no: int) -> None:
        self.rejected[reason] += 1
        s = self.rejected_samples.setdefault(reason, [])
        if len(s) < MAX_SAMPLES_PER_REASON:
            s.append(line_no)

    def summary(self) -> dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "accepted": len(self.requests),
            "rejected_total": self.rejected_total,
            "rejected_by_reason": dict(sorted(self.rejected.items())),
            "rejected_sample_lines": {k: v for k, v in sorted(self.rejected_samples.items())},
            "replay_order": self.order,
        }


class ValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _parse_timestamp(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValidationError("invalid_timestamp")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            raise ValidationError("invalid_timestamp") from None
    raise ValidationError("invalid_timestamp")


def _validate_message(m: Any) -> None:
    if not isinstance(m, dict):
        raise ValidationError("message_not_object")
    role = m.get("role")
    if role not in VALID_ROLES:
        raise ValidationError("invalid_role")
    content = m.get("content")
    if content is None:
        if role != "assistant" or not m.get("tool_calls"):
            raise ValidationError("missing_content")
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict) or "type" not in part:
                raise ValidationError("invalid_content_part")
    elif not isinstance(content, str):
        raise ValidationError("invalid_content_type")
    tcs = m.get("tool_calls")
    if tcs is not None:
        if role != "assistant" or not isinstance(tcs, list):
            raise ValidationError("invalid_tool_calls")
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValidationError("invalid_tool_calls")


def validate_obj(obj: Any) -> tuple[str, list[dict], list[dict] | None, str | None, float | None, str | None]:
    if not isinstance(obj, dict):
        raise ValidationError("not_an_object")
    model = obj.get("model")
    if not isinstance(model, str) or not model:
        raise ValidationError("missing_model")
    messages = obj.get("messages")
    if not isinstance(messages, list):
        raise ValidationError("missing_messages")
    if not messages:
        raise ValidationError("empty_messages")
    for m in messages:
        _validate_message(m)
    tools = obj.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            raise ValidationError("invalid_tools")
        if not tools:
            tools = None
    sid = obj.get("session_id")
    if sid is not None:
        if isinstance(sid, bool) or not isinstance(sid, (str, int)):
            raise ValidationError("invalid_session_id")
        sid = str(sid) if str(sid) != "" else None
    ts = _parse_timestamp(obj.get("timestamp"))
    rid = obj.get("request_id")
    rid = str(rid) if rid is not None else None
    return model, messages, tools, sid, ts, rid


def load_jsonl(path: str, order: str = "file") -> IngestResult:
    """Load a JSONL request log.

    ``order``: ``"file"`` replays in file order (default; assumes the log is
    written in arrival order). ``"timestamp"`` stable-sorts by timestamp;
    requests without a timestamp keep their relative file position at the end.
    """
    with open(path, encoding="utf-8") as f:
        return load_lines(f, order=order)


def load_lines(lines, order: str = "file") -> IngestResult:
    if order not in ("file", "timestamp"):
        raise ValueError("order must be 'file' or 'timestamp'")
    res = IngestResult(order=order)
    staged: list[Request] = []
    for line_no, line in enumerate(lines, start=1):
        res.total_lines += 1
        if not line.strip():
            res.blank_lines += 1
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            res.reject("invalid_json", line_no)
            continue
        try:
            model, messages, tools, sid, ts, rid = validate_obj(obj)
        except ValidationError as e:
            res.reject(e.reason, line_no)
            continue
        staged.append(Request(-1, line_no, model, messages, tools, sid, ts, rid))
    if order == "timestamp":
        staged.sort(key=lambda r: (r.timestamp is None, r.timestamp or 0.0, r.line_no))
    for i, r in enumerate(staged):
        r.index = i
    res.requests = staged
    return res
