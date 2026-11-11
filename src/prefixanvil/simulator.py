"""Block-level prefix-cache simulator.

A *specification-based approximation* of vLLM V1 Automatic Prefix Caching
(APC). It is NOT vLLM and has not been validated against a real engine (that
is Phase 3). What it models, and where it deliberately differs, is listed in
README.md ("Simulator model"). In short:

* Prompts are split into fixed-size blocks (default 16 tokens). Only *full*
  blocks are hashed and cacheable; a trailing partial block is computed and
  never reused.
* Block hash = H(parent_hash, token_ids_of_block[, extra_key]) — a chain, so a
  block can only hit if every preceding block hit. ``extra_key`` optionally
  carries the model name (vLLM itself keys on LoRA / multimodal / cache_salt,
  not on model name; one engine serves one model).
* Lookup stops at the first miss. Like vLLM V1, at least the last prompt token
  is always recomputed (logits are needed), so the hit length is capped at
  ``num_tokens - 1`` rounded down to a block boundary.
* Fixed pool of ``num_blocks`` physical blocks. Free blocks live in an LRU
  queue; allocation pops from the front and evicts any cached hash it held.
  When a request finishes, its blocks are appended to the queue *in reverse
  order* (tail blocks are evicted before the head), matching vLLM V1.
* Requests are processed strictly one at a time (no concurrency, no batching,
  no chunked-prefill interleaving) and decode tokens are not simulated.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass, field

# historical constant, deliberately kept across the PrefixForge -> PrefixAnvil rename so that
# block hashes (and the golden value in tests/test_simulator.py) stay identical
ROOT_SEED = b"prefixforge/v1/root"


def block_hashes(token_ids: list[int], block_size: int, extra_key: str | None = None) -> list[bytes]:
    """Chained hashes for every *full* block of ``token_ids``."""
    out: list[bytes] = []
    parent = hashlib.sha256(ROOT_SEED).digest()
    extra = (extra_key or "").encode("utf-8")
    n_full = len(token_ids) // block_size
    for b in range(n_full):
        chunk = token_ids[b * block_size:(b + 1) * block_size]
        h = hashlib.sha256()
        h.update(parent)
        h.update(struct.pack(f"<{len(chunk)}q", *chunk))
        h.update(struct.pack("<I", len(extra)))
        h.update(extra)
        parent = h.digest()
        out.append(parent)
    return out


@dataclass
class RequestResult:
    index: int
    prompt_tokens: int
    cached_tokens: int
    prefill_tokens: int
    first_miss_block: int | None    # index of the first block that was not reused (None: no miss)
    evictions: int
    oversize: bool = False


@dataclass
class SimStats:
    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefill_tokens: int = 0
    evictions: int = 0
    oversize_requests: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class SimConfig:
    block_size: int = 16
    num_blocks: int = 8192
    hash_model_name: bool = False

    def as_dict(self) -> dict:
        return {"block_size": self.block_size, "num_blocks": self.num_blocks,
                "capacity_tokens": self.block_size * self.num_blocks,
                "hash_model_name": self.hash_model_name}


class PrefixCacheSimulator:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        if self.cfg.block_size < 1 or self.cfg.num_blocks < 1:
            raise ValueError("block_size and num_blocks must be >= 1")
        n = self.cfg.num_blocks
        self._block_hash: list[bytes | None] = [None] * n
        self._free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(n))
        self._cached: dict[bytes, int] = {}
        self.stats = SimStats()
        self.results: list[RequestResult] = []

    # -- introspection helpers (tests) -------------------------------------
    def is_cached(self, h: bytes) -> bool:
        return h in self._cached

    @property
    def cached_block_count(self) -> int:
        return len(self._cached)

    # ----------------------------------------------------------------------
    def _alloc(self) -> tuple[int, bool]:
        bid, _ = self._free.popitem(last=False)
        evicted = False
        h = self._block_hash[bid]
        if h is not None:
            if self._cached.get(h) == bid:
                del self._cached[h]
                evicted = True
            self._block_hash[bid] = None
        return bid, evicted

    def process(self, token_ids: list[int], model: str | None = None, index: int | None = None) -> RequestResult:
        B = self.cfg.block_size
        n = len(token_ids)
        idx = self.stats.requests if index is None else index
        extra = model if self.cfg.hash_model_name else None
        hashes = block_hashes(token_ids, B, extra)
        n_blocks_total = -(-n // B)

        if n_blocks_total > self.cfg.num_blocks:
            res = RequestResult(idx, n, 0, n, None, 0, oversize=True)
            self._account(res)
            return res

        max_hit_blocks = max(0, (n - 1) // B) if n else 0
        hit_ids: list[int] = []
        for i in range(max_hit_blocks):
            bid = self._cached.get(hashes[i])
            if bid is None:
                break
            hit_ids.append(bid)
        for bid in hit_ids:             # touch: in use -> not evictable
            self._free.pop(bid, None)

        evictions = 0
        new_ids: list[int] = []
        for i in range(len(hit_ids), n_blocks_total):
            bid, ev = self._alloc()
            evictions += ev
            new_ids.append(bid)
            if i < len(hashes):          # full block -> becomes cacheable
                h = hashes[i]
                self._block_hash[bid] = h
                self._cached[h] = bid
        # request finishes: free in reverse order so tail blocks are evicted first
        for bid in reversed(hit_ids + new_ids):
            self._free[bid] = None

        cached = len(hit_ids) * B
        # first block whose lookup failed; None if every lookup-eligible block hit
        first_miss = len(hit_ids) if len(hit_ids) < max_hit_blocks else None
        res = RequestResult(idx, n, cached, n - cached, first_miss, evictions)
        self._account(res)
        return res

    def _account(self, res: RequestResult) -> None:
        s = self.stats
        s.requests += 1
        s.prompt_tokens += res.prompt_tokens
        s.cached_tokens += res.cached_tokens
        s.prefill_tokens += res.prefill_tokens
        s.evictions += res.evictions
        s.oversize_requests += res.oversize
        self.results.append(res)


def replay(token_lists: list[list[int]], cfg: SimConfig | None = None,
           models: list[str] | None = None) -> PrefixCacheSimulator:
    sim = PrefixCacheSimulator(cfg)
    for i, toks in enumerate(token_lists):
        sim.process(toks, models[i] if models else None, index=i)
    return sim


def aligned(n: int, block_size: int) -> int:
    return (n // block_size) * block_size


def max_reusable(prompt_len: int, block_size: int) -> int:
    """Upper limit of reusable tokens for one prompt (last token always recomputed)."""
    return aligned(max(0, prompt_len - 1), block_size)
