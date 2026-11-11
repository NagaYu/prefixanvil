"""Simulator fidelity: replay the *same* trace through the block simulator with the
real tokenizer, real block size and real KV capacity, and compare against what
vLLM measured.

Two rendering assumptions are simulated side by side:
  * ``client``: the client's bytes (Phase 1/2 simulator behaviour)
  * ``vllm``:   vLLM 0.33.0's server-side normalization of tools / tool-call
                arguments before templating (see ``prefixanvil.render.vllm_normalize``)
Policies match the bench conditions: ``raw``, ``L1`` (session-aware tool order,
like the proxy with a session header), ``L1-stateless`` (Phase-1 name sort).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from prefixanvil.canonical import SessionContext, canonicalize, next_session_context, sort_keys_recursive
from prefixanvil.ingest import load_lines
from prefixanvil.render import Renderer
from prefixanvil.simulator import PrefixCacheSimulator, SimConfig

POLICIES = ("raw", "L1", "L1-stateless")


@dataclass
class SimRun:
    policy: str
    rendering: str
    prompt_tokens: list[int]
    cached_tokens: list[int]

    @property
    def totals(self) -> dict:
        p, c = sum(self.prompt_tokens), sum(self.cached_tokens)
        return {"prompt_tokens": p, "cached_tokens": c, "prefill_tokens": p - c,
                "hit_rate": c / p if p else None}


def _requests(trace_path: str, limit: int | None):
    with open(trace_path, encoding="utf-8") as f:
        lines = [x for x in f if x.strip()]
    return load_lines(lines[:limit] if limit else lines).requests


def simulate(trace_path: str, tok, policy: str, rendering: str, block_size: int, num_blocks: int,
             limit: int | None = None) -> SimRun:
    if policy not in POLICIES or rendering not in ("client", "vllm"):
        raise ValueError((policy, rendering))
    renderer = Renderer(tok, "vllm" if rendering == "vllm" else None)
    sim = PrefixCacheSimulator(SimConfig(block_size, num_blocks))
    ctxs: dict[str, SessionContext] = {}
    pt, ct = [], []
    for req in _requests(trace_path, limit):
        if policy != "raw":
            ctx = None
            if policy == "L1" and req.session_id:
                ctx = ctxs.get(req.session_id, SessionContext())
            req2, _ = canonicalize(req, ctx)
            if ctx is not None:
                ctxs[req.session_id] = next_session_context(req2)
            req = req2
        toks = renderer.render(req).token_ids
        r = sim.process(toks)
        pt.append(r.prompt_tokens)
        ct.append(r.cached_tokens)
    return SimRun(policy, rendering, pt, ct)


def token_change(trace_path: str, tok, limit: int | None = None) -> dict:
    """Prompt-token change from L1 (whole) and from tools key sorting alone, under both renderings."""
    reqs = _requests(trace_path, limit)
    out = {}
    for rendering in ("client", "vllm"):
        r = Renderer(tok, "vllm" if rendering == "vllm" else None)
        raw = keysort = l1 = 0
        for q in reqs:
            raw += len(r.render(q).token_ids)
            ks = load_lines([json.dumps({"model": q.model, "messages": q.messages,
                                         "tools": [sort_keys_recursive(t) for t in q.tools]
                                         if q.tools else None})]).requests[0]
            keysort += len(r.render(ks).token_ids)
            l1 += len(r.render(canonicalize(q)[0]).token_ids)
        out[rendering] = {"raw_tokens": raw, "keysort_tokens": keysort, "l1_tokens": l1,
                          "keysort_change": (keysort - raw) / raw if raw else None,
                          "l1_change": (l1 - raw) / raw if raw else None}
    return out


def measured_pass(cond_json: dict, n: int = 1) -> dict:
    """Totals of measured pass n of a condition record (bench.runner output)."""
    p = [x for x in cond_json["passes"] if x["n"] == n][0]
    c = p["counters"]
    res = [r for r in p["results"] if r.get("error") is None]
    return {
        "hits": c.get("prefix_cache_hits"), "queries": c.get("prefix_cache_queries"),
        "hit_rate": c.get("hit_rate"),
        "prefill_tokens_counter": (c["prefix_cache_queries"] - c["prefix_cache_hits"])
        if c.get("prefix_cache_queries") is not None and c.get("prefix_cache_hits") is not None else None,
        "prefill_kv_computed_sum": c.get("prefill_kv_computed_sum"),
        "per_request_prompt": [r.get("prompt_tokens") for r in sorted(res, key=lambda r: r["index"])],
        "per_request_cached": [r.get("cached_tokens") for r in sorted(res, key=lambda r: r["index"])],
    }


def compare(measured: dict, sim: SimRun, block_size: int) -> dict:
    t = sim.totals
    row = {"policy": sim.policy, "rendering": sim.rendering,
           "sim_hit_rate": t["hit_rate"], "real_hit_rate": measured["hit_rate"],
           "sim_prefill_tokens": t["prefill_tokens"], "real_prefill_tokens": measured["prefill_tokens_counter"]}
    if row["real_hit_rate"] is not None and t["hit_rate"] is not None:
        row["hit_rate_diff_pts"] = (t["hit_rate"] - row["real_hit_rate"]) * 100
    if row["real_prefill_tokens"]:
        row["prefill_diff_pct"] = (t["prefill_tokens"] - row["real_prefill_tokens"]) / row["real_prefill_tokens"] * 100
    rp, rc = measured["per_request_prompt"], measured["per_request_cached"]
    if rp and len(rp) == len(sim.prompt_tokens) and all(x is not None for x in rp):
        row["prompt_tokens_exact"] = sum(a == b for a, b in zip(rp, sim.prompt_tokens))
        row["prompt_tokens_total_real"] = sum(rp)
        row["prompt_tokens_total_sim"] = sum(sim.prompt_tokens)
        row["requests"] = len(rp)
    if rc and len(rc) == len(sim.cached_tokens) and all(x is not None for x in rc):
        d = [s - r for s, r in zip(sim.cached_tokens, rc)]
        row["cached_exact"] = sum(x == 0 for x in d)
        row["cached_within_1_block"] = sum(abs(x) <= block_size for x in d)
        row["cached_sim_over_real_requests"] = sum(x > 0 for x in d)
        row["cached_sim_under_real_requests"] = sum(x < 0 for x in d)
    return row
