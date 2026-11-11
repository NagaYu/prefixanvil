"""Session-order-preserving replay of AgentLoad traces against an OpenAI-compatible server.

Pure part: :func:`build_items` turns trace lines into request bodies
(temperature 0, fixed ``max_tokens``, streaming with usage) and computes each
request's in-session predecessor. Thin part: :func:`replay` sends them with
concurrency 1 (deterministic cache accounting: one request at a time, in file
order) or N (TTFT reference: up to N in flight, a request never starts before
its session predecessor finished).

The replay is open-loop: later turns use the assistant messages recorded in the
trace, not what the model generated, so decode-phase KV of the model's own
answers is never reused (same assumption as the simulator).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable
from urllib.parse import urlsplit

_DROP = ("request_id", "session_id", "timestamp", "metadata")


@dataclass
class ReplayItem:
    index: int
    request_id: str | None
    session_id: str | None
    predecessor: int | None
    body: dict


def build_items(lines: list[dict], served_model: str, max_tokens: int = 64,
                limit: int | None = None, seed: int = 0) -> list[ReplayItem]:
    items, last = [], {}
    for i, rec in enumerate(lines[:limit] if limit else lines):
        body = {k: v for k, v in rec.items() if k not in _DROP}
        body.update(model=served_model, temperature=0, max_tokens=max_tokens, seed=seed,
                    stream=True, stream_options={"include_usage": True})
        sid = rec.get("session_id")
        items.append(ReplayItem(i, rec.get("request_id"), sid, last.get(sid) if sid else None, body))
        if sid:
            last[sid] = i
    return items


def load_trace(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


@dataclass
class Result:
    index: int
    request_id: str | None
    session_id: str | None
    status: int
    ttft_s: float | None
    total_s: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    output_sha256: str | None = None
    error: str | None = None
    started_at: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def send_streaming(base_url: str, body: dict, headers: dict | None = None,
                   timeout: float = 600.0, keep_text: bool = False) -> tuple[Result, str | None]:
    """POST one streaming chat completion; measure TTFT (first content/tool delta)."""
    u = urlsplit(base_url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port, timeout=timeout)
    t0 = time.perf_counter()
    res = Result(-1, None, None, 0, None, 0.0, started_at=time.time())
    text_parts: list[str] = []
    try:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn.request("POST", u.path.rstrip("/") + "/v1/chat/completions", data,
                     {"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        res.status = resp.status
        if resp.status != 200:
            res.error = f"http_{resp.status}:{resp.read()[:200].decode('utf-8', 'replace')}"
            return res, None
        buf = b""
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        piece += (fn.get("name") or "") + (fn.get("arguments") or "")
                    if piece:
                        if res.ttft_s is None:
                            res.ttft_s = time.perf_counter() - t0
                        text_parts.append(piece)
                usage = ev.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens")
                    res.completion_tokens = usage.get("completion_tokens")
                    det = usage.get("prompt_tokens_details") or {}
                    res.cached_tokens = det.get("cached_tokens")
    except Exception as e:  # recorded, never raised: a failed request must not kill a pass
        res.error = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
        res.total_s = time.perf_counter() - t0
    text = "".join(text_parts)
    res.output_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return res, (text if keep_text else None)


def replay(items: list[ReplayItem], base_url: str, concurrency: int = 1,
           session_header: bool = True, send: Callable = send_streaming,
           on_result: Callable[[Result], None] | None = None) -> list[Result]:
    """Send all items. concurrency=1: strictly sequential in file order."""
    results: dict[int, Result] = {}
    done = {it.index: threading.Event() for it in items}
    sem = threading.Semaphore(max(1, concurrency))
    lock = threading.Lock()

    def run(it: ReplayItem) -> None:
        try:
            hdr = {"X-PrefixAnvil-Session-Id": it.session_id} if session_header and it.session_id else {}
            r, _ = send(base_url, it.body, hdr)
            r.index, r.request_id, r.session_id = it.index, it.request_id, it.session_id
            with lock:
                results[it.index] = r
            if on_result:
                on_result(r)
        finally:
            done[it.index].set()
            sem.release()

    threads = []
    for it in items:
        if it.predecessor is not None:
            done[it.predecessor].wait()
        sem.acquire()
        if concurrency <= 1:
            run(it)
        else:
            t = threading.Thread(target=run, args=(it,), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return [results[i.index] for i in items if i.index in results]


@dataclass
class PassSummary:
    requests: int
    errors: int
    wall_s: float
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    per_request_cached_available: bool
    extra: dict = field(default_factory=dict)


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[Result], wall_s: float) -> PassSummary:
    ok = [r for r in results if r.error is None and r.status == 200]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    have_cached = bool(ok) and all(r.cached_tokens is not None for r in ok)
    have_prompt = bool(ok) and all(r.prompt_tokens is not None for r in ok)
    return PassSummary(
        requests=len(results), errors=len(results) - len(ok), wall_s=wall_s,
        ttft_p50_s=percentile(ttfts, 0.50), ttft_p95_s=percentile(ttfts, 0.95),
        prompt_tokens=sum(r.prompt_tokens for r in ok) if have_prompt else None,
        cached_tokens=sum(r.cached_tokens for r in ok) if have_cached else None,
        per_request_cached_available=have_cached)
