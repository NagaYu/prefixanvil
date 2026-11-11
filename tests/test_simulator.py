"""Hand-computed cases for the block simulator (block_size=4 to keep arithmetic small)."""

from prefixanvil.simulator import (PrefixCacheSimulator, SimConfig, block_hashes, max_reusable,
                                   replay)

B = 4


def sim(num_blocks=64, **kw):
    return PrefixCacheSimulator(SimConfig(block_size=B, num_blocks=num_blocks, **kw))


def test_cold_then_exact_match():
    s = sim()
    toks = list(range(12))                 # 3 full blocks
    r1 = s.process(toks)
    assert (r1.prompt_tokens, r1.cached_tokens, r1.prefill_tokens) == (12, 0, 12)
    assert r1.first_miss_block == 0
    r2 = s.process(toks)
    # last token must be recomputed -> only (12-1)//4 = 2 blocks can hit
    assert (r2.cached_tokens, r2.prefill_tokens) == (8, 4)
    assert r2.first_miss_block is None


def test_partial_hit_stops_at_first_miss():
    s = sim()
    s.process(list(range(12)))
    c = [0, 1, 2, 3, 4, 5, 99, 7, 8, 9, 10, 11]   # block 1 differs at token 6
    r = s.process(c)
    assert r.cached_tokens == 4 and r.first_miss_block == 1
    # block 2 equals the cached block 2 content, but its parent differs -> no hit
    assert r.prefill_tokens == 8


def test_hash_chain_depends_on_parent():
    h = block_hashes([1, 2, 3, 4, 1, 2, 3, 4], B)
    assert len(h) == 2 and h[0] != h[1]
    s = sim()
    s.process([1, 2, 3, 4, 5])
    r = s.process([9, 9, 9, 9, 1, 2, 3, 4, 5])
    assert r.cached_tokens == 0


def test_block_boundary_partial_blocks_are_not_cached():
    s = sim()
    s.process(list(range(10)))             # 2 full blocks + partial [8, 9]
    r = s.process(list(range(14)))         # needs block 2 = [8..11]: never cached
    assert r.cached_tokens == 8 and r.first_miss_block == 2


def test_last_token_rule_at_boundaries():
    s = sim()
    s.process(list(range(12)))
    assert s.process(list(range(9))).cached_tokens == 8     # (9-1)//4 = 2 blocks
    assert s.process(list(range(8))).cached_tokens == 4     # (8-1)//4 = 1 block
    assert s.process(list(range(1))).cached_tokens == 0
    assert max_reusable(8, B) == 4 and max_reusable(9, B) == 8 and max_reusable(0, B) == 0


def test_lru_eviction_keeps_recently_used():
    # capacity 4 blocks; each 5-token request uses 2 blocks (1 cacheable + 1 partial)
    s = sim(num_blocks=4)
    X, Y, Z = [1] * 5, [2] * 5, [3] * 5
    s.process(X)
    s.process(Y)
    assert s.process(X).cached_tokens == 4          # touch X -> most recently used
    rz = s.process(Z)                               # must evict: LRU victim is Y, not X
    assert rz.evictions == 1
    assert s.process(Y).cached_tokens == 0
    s2 = sim(num_blocks=4)
    for t in (X, Y, X, Z):
        s2.process(t)
    assert s2.process(X).cached_tokens == 4


def test_eviction_frees_tail_blocks_first():
    s = sim(num_blocks=3)
    X = list(range(12))                    # 3 full blocks, occupies whole cache
    s.process(X)
    s.process([100, 101, 102, 103])        # 1 block: evicts X's *last* block
    r = s.process(X)
    assert r.cached_tokens == 8            # head blocks survived (vLLM frees in reverse order)


def test_oversize_request_is_counted_not_cached():
    s = sim(num_blocks=2)
    r = s.process(list(range(20)))
    assert r.oversize and r.cached_tokens == 0 and r.prefill_tokens == 20
    assert s.stats.oversize_requests == 1


def test_model_name_in_hash_is_optional():
    toks = list(range(9))
    s = sim(hash_model_name=True)
    s.process(toks, model="a")
    assert s.process(toks, model="b").cached_tokens == 0
    assert s.process(toks, model="a").cached_tokens == 8
    s2 = sim()
    s2.process(toks, model="a")
    assert s2.process(toks, model="b").cached_tokens == 8


def test_deterministic_and_stable_hash():
    seqs = [list(range(i, i + 37)) for i in range(0, 50, 7)] * 3
    a = replay(seqs, SimConfig(B, 16))
    b = replay(seqs, SimConfig(B, 16))
    assert [vars(r) for r in a.results] == [vars(r) for r in b.results]
    # golden value: hashing is process-independent (no PYTHONHASHSEED dependence)
    assert block_hashes([1, 2, 3, 4], 4)[0].hex()[:16] == GOLDEN


def test_stats_aggregate():
    s = sim()
    s.process(list(range(12)))
    s.process(list(range(12)))
    assert s.stats.prompt_tokens == 24 and s.stats.cached_tokens == 8
    assert abs(s.stats.hit_rate - 8 / 24) < 1e-12


GOLDEN = "9783ca97a1c7e122"
