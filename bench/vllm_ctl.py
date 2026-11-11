"""Thin HTTP control of a vLLM OpenAI server: health, /metrics, prefix-cache reset.

``POST /reset_prefix_cache`` exists in vLLM 0.33.0 only when the server runs
with ``VLLM_SERVER_DEV_MODE=1`` (vllm/entrypoints/serve/dev/cache/api_router.py)
and returns ``{"success": bool}``; it fails while blocks are held by running
requests, so callers retry. Prometheus counters are NOT reset by it (they are
monotonic), which is why every measurement uses counter deltas, and why reset
success is verified behaviourally (:func:`verify_reset`).
"""

from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlsplit

from .metrics import diff, snapshot
from .replay import send_streaming


def _req(base: str, method: str, path: str, body: bytes | None = None, timeout: float = 30.0):
    u = urlsplit(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        c.request(method, u.path.rstrip("/") + path, body,
                  {"Content-Type": "application/json"} if body is not None else {})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def wait_healthy(base: str, timeout_s: float = 1800.0, poll_s: float = 5.0, alive=lambda: True) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not alive():
            return False
        try:
            st, _ = _req(base, "GET", "/health", timeout=5)
            if st == 200:
                return True
        except OSError:
            pass
        time.sleep(poll_s)
    return False


def metrics_text(base: str) -> str:
    st, body = _req(base, "GET", "/metrics")
    if st != 200:
        raise RuntimeError(f"/metrics returned {st}")
    return body.decode("utf-8", "replace")


def reset_prefix_cache(base: str, retries: int = 10, wait_s: float = 2.0) -> dict:
    """Returns {"status": "ok"|"unsupported"|"failed", "attempts": n}."""
    for i in range(1, retries + 1):
        try:
            st, body = _req(base, "POST", "/reset_prefix_cache", b"")
        except OSError as e:
            return {"status": "failed", "attempts": i, "error": type(e).__name__}
        if st == 404:
            return {"status": "unsupported", "attempts": i}
        if st == 200:
            try:
                ok = json.loads(body or b"{}").get("success", True)
            except json.JSONDecodeError:
                ok = True  # older versions returned an empty 200
            if ok:
                return {"status": "ok", "attempts": i}
        time.sleep(wait_s)
    return {"status": "failed", "attempts": retries}


PROBE_TEXT = " ".join(f"probe-sentence {i} about deterministic prefix cache verification." for i in range(120))


def verify_reset(base: str, served_model: str) -> dict:
    """Behavioural check: a repeated probe hits the cache, and misses right after a reset.

    Leaves the cache reset (the probe's blocks are cleared at the end).
    """
    body = {"model": served_model, "messages": [{"role": "user", "content": PROBE_TEXT}],
            "temperature": 0, "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True}}
    out: dict = {}
    s0 = snapshot(metrics_text(base))
    send_streaming(base, body)
    send_streaming(base, body)
    s1 = snapshot(metrics_text(base))
    out["warm_hits"] = diff(s0, s1)["prefix_cache_hits"]
    out["reset_1"] = reset_prefix_cache(base)
    s2 = snapshot(metrics_text(base))
    send_streaming(base, body)
    s3 = snapshot(metrics_text(base))
    out["hits_after_reset"] = diff(s2, s3)["prefix_cache_hits"]
    out["reset_2"] = reset_prefix_cache(base)
    out["verified"] = (out["reset_1"]["status"] == "ok" and out["reset_2"]["status"] == "ok"
                       and bool(out["warm_hits"]) and out["hits_after_reset"] == 0)
    return out
