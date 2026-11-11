"""Prometheus text parsing and search-based resolution of vLLM metric names.

Metric names drift between vLLM versions (e.g. ``vllm:gpu_prefix_cache_hits``
-> ``vllm:prefix_cache_hits``; counters gain ``_total`` in the exposition), so
logical metrics are resolved by regex search over the scraped text instead of
hard-coded names. Anything that cannot be resolved is reported as missing and
the raw text is kept by the caller.

Verified against vLLM 0.33.0 source (vllm/v1/metrics/loggers.py):
``vllm:prefix_cache_queries`` / ``vllm:prefix_cache_hits`` (counters, in
tokens), ``vllm:prompt_tokens`` / ``vllm:prompt_tokens_cached`` (counters),
``vllm:request_prefill_kv_computed_tokens`` (histogram: new KV tokens computed
during prefill, excluding cached), ``vllm:cache_config_info`` (gauge whose
labels carry the CacheConfig fields, incl. ``block_size``, ``num_gpu_blocks``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# logical name -> ordered regex candidates (first match wins)
COUNTERS: dict[str, list[str]] = {
    "prefix_cache_hits": [r"vllm:prefix_cache_hits(_total)?", r"vllm:gpu_prefix_cache_hits(_total)?"],
    "prefix_cache_queries": [r"vllm:prefix_cache_queries(_total)?", r"vllm:gpu_prefix_cache_queries(_total)?"],
    "prefill_kv_computed_sum": [r"vllm:request_prefill_kv_computed_tokens_sum"],
    "prefill_kv_computed_count": [r"vllm:request_prefill_kv_computed_tokens_count"],
    "prompt_tokens": [r"vllm:prompt_tokens(_total)?"],
    "prompt_tokens_cached": [r"vllm:prompt_tokens_cached(_total)?"],
    "generation_tokens": [r"vllm:generation_tokens(_total)?"],
    "request_success": [r"vllm:request_success(_total)?"],
    "preemptions": [r"vllm:num_preemptions(_total)?"],
}
INFO_GAUGES = {"cache_config": [r"vllm:cache_config_info"]}

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+([^\s]+)(\s+\d+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse_prometheus(text: str) -> list[Sample]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, _, labels, value, _ = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        lab = tuple(sorted((k, val.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\"))
                           for k, val in _LABEL.findall(labels or "")))
        out.append(Sample(name, lab, v))
    return out


def _resolve(samples: list[Sample], patterns: list[str]) -> tuple[str | None, list[Sample]]:
    names = sorted({s.name for s in samples})
    for p in patterns:
        rx = re.compile(p + r"$")
        hit = [n for n in names if rx.match(n)]
        if hit:
            n = hit[0]
            return n, [s for s in samples if s.name == n]
    return None, []


def snapshot(text: str) -> dict:
    """Resolve logical metrics from one scrape. Values are summed over label sets
    (multiple engines, finished_reason, ...)."""
    samples = parse_prometheus(text)
    values, names, missing = {}, {}, []
    for logical, pats in COUNTERS.items():
        n, ss = _resolve(samples, pats)
        if n is None:
            missing.append(logical)
            values[logical] = None
        else:
            names[logical] = n
            values[logical] = sum(s.value for s in ss)
    info = {}
    for logical, pats in INFO_GAUGES.items():
        n, ss = _resolve(samples, pats)
        if ss:
            info[logical] = dict(ss[0].labels)
    return {"values": values, "names": names, "missing": missing, "info": info}


def diff(before: dict, after: dict) -> dict:
    """Counter deltas between two snapshots (None if missing in either)."""
    out = {}
    for k in COUNTERS:
        a, b = before["values"].get(k), after["values"].get(k)
        out[k] = None if a is None or b is None else b - a
    hits, queries = out.get("prefix_cache_hits"), out.get("prefix_cache_queries")
    out["hit_rate"] = (hits / queries) if hits is not None and queries else None
    return out


def cache_config(snap: dict) -> dict:
    """block_size / num_gpu_blocks / prefix caching flags from vllm:cache_config_info."""
    info = snap.get("info", {}).get("cache_config", {})

    def as_int(k):
        try:
            return int(float(info[k]))
        except (KeyError, ValueError):
            return None
    return {"block_size": as_int("block_size"), "num_gpu_blocks": as_int("num_gpu_blocks"),
            "hash_block_size": as_int("hash_block_size"),
            "enable_prefix_caching": info.get("enable_prefix_caching"),
            "prefix_caching_hash_algo": info.get("prefix_caching_hash_algo"),
            "raw": info}
