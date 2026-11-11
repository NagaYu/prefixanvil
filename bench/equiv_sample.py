"""Real-model equivalence sampler: replay the Phase 2 equiv fixtures against a real server.

Per fixture request (temperature 0, fixed max_tokens and seed):
  raw#1 (cold cache)  vs  raw#2 (cold cache)      -> control: self-consistency
  raw#1               vs  L1 (cold cache)         -> effect of L1
  raw#1               vs  L2 (cold cache)         -> effect of L2 (cases that enable L2)
  raw#1               vs  raw#3 (warm cache)      -> effect of prefix-cache hits themselves
The cache is reset before every cold send, so canonicalization effects are not
confounded with cached-vs-recomputed numerics; the warm send measures that
confound separately. L1/L2 rates should be judged *relative to the control*.

Storage: output hashes always; full output texts only for mismatching pairs.
Mismatches get an automatic first-pass tag (``formatting_only`` if equal after
case/whitespace/punctuation folding, else ``needs_review``); deciding whether a
difference changes meaning is a human step (BENCH.md).
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Callable

from prefixanvil.core import TRANSFORM_REQUEST_SCHEMA, transform
from prefixanvil.equiv import load_suite

from .replay import send_streaming
from .vllm_ctl import reset_prefix_cache

SAMPLE_SCHEMA = "prefixanvil.bench.equiv-sample/v1"


def _fold(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.casefold()).strip()


def _first_diff(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def _pair(a: str, b: str) -> dict:
    if a == b:
        return {"match": True}
    return {"match": False, "first_diff_char": _first_diff(a, b),
            "auto_tag": "formatting_only" if _fold(a) == _fold(b) else "needs_review",
            "text_a": a, "text_b": b}


def run(suite_dir: str, base_url: str, served_model: str, max_tokens: int = 64, seed: int = 0,
        send: Callable = send_streaming, reset: Callable = reset_prefix_cache) -> dict:
    cases, suite_hash = load_suite(suite_dir)
    rows = []

    def call(body: dict, cold: bool) -> str:
        if cold:
            r = reset(base_url)
            if r["status"] != "ok":
                raise RuntimeError(f"cache reset failed: {r}")
        b = copy.deepcopy(body)
        b.update(model=served_model, temperature=0, max_tokens=max_tokens, seed=seed, stream=True,
                 stream_options={"include_usage": True})
        res, text = send(base_url, b, None, keep_text=True)
        if res.error:
            raise RuntimeError(res.error)
        return text or ""

    for c in cases:
        ctx = {"L1": {"schema": "prefixanvil.session-context/v1", "prev_tool_names": None, "prev_tool_order": None},
               "L2": {"schema": "prefixanvil.session-context/v1", "prev_tool_names": None, "prev_tool_order": None}}
        for k, body in enumerate(c["requests"]):
            row = {"case": c["id"], "index": k}
            try:
                raw1 = call(body, cold=True)
                raw2 = call(body, cold=True)
                raw3 = call(body, cold=False)
                row["control"] = _pair(raw1, raw2)
                row["warm_cache"] = _pair(raw1, raw3)
                row["raw_sha256"] = hashlib.sha256(raw1.encode()).hexdigest()[:16]
                for mode in ("L1", "L2"):
                    if mode not in c["expect"]:
                        continue
                    res = transform({"schema": TRANSFORM_REQUEST_SCHEMA, "mode": mode, "request": body,
                                     "session_context": ctx[mode] if c.get("session", True) else None})
                    ctx[mode] = res["session_context"]
                    if not res["changed"]:
                        row[mode] = {"match": True, "unchanged_request": True}
                        continue
                    row[mode] = _pair(raw1, call(res["request"], cold=True))
                    row[mode]["rules"] = sorted({a["rule"] for a in res["applied"]})
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)

    def rate(key: str, changed_only: bool = False) -> dict:
        xs = [r[key] for r in rows if key in r and not (changed_only and r[key].get("unchanged_request"))]
        m = sum(x["match"] for x in xs)
        return {"pairs": len(xs), "match": m, "rate": m / len(xs) if xs else None}

    return {"schema": SAMPLE_SCHEMA, "suite": {"path": suite_dir, "sha256": suite_hash},
            "served_model": served_model, "max_tokens": max_tokens, "seed": seed,
            "summary": {"control_raw_vs_raw": rate("control"), "warm_cache_vs_cold": rate("warm_cache"),
                        "L1_vs_raw": rate("L1"), "L1_vs_raw_changed_only": rate("L1", True),
                        "L2_vs_raw": rate("L2"), "L2_vs_raw_changed_only": rate("L2", True),
                        "errors": sum("error" in r for r in rows)},
            "rows": rows}
