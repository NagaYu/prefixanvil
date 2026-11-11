# Phase 3 pre-registered simulator predictions

Written on the development Mac **before any GPU run** (git history is the timestamp). The Phase 3
fidelity table compares the vLLM measurements against these numbers *and* against a re-run of the
simulator with the real server's block size and KV capacity (`fidelity.json` in the results). Nothing
here is a measurement.

- Tokenizer + chat template: `Qwen/Qwen2.5-7B-Instruct-AWQ` @ `b25037543e9394b818fdfca67ab2a00ecc7dd641`
  (tokenizer files only; license `apache-2.0` per the Hub card), template `sha256:cd8e9439f0570856`,
  no `strftime_now` in the template.
- Block size 16 (vLLM 0.33.0 `CacheConfig.DEFAULT_BLOCK_SIZE`; the real value is read from
  `vllm:cache_config_info` on the server). KV capacity: no eviction, and an estimated 8,000 blocks for a
  T4 (16 GB − ~5.2 GB AWQ weights, 0.90 utilization, 57,344 B/token for Qwen2.5-7B fp16 KV) — the estimate
  changed nothing on these traces.
- Rendering: `client` = the bytes the client sends (Phase 1/2 simulator); `vllm` = vLLM 0.33.0's
  server-side normalization (tools via `model_dump()`, `arguments` parsed to dicts).
- Raw data: `fixtures/bench/preregistered_predictions.json`.

## Traces (regenerated deterministically from `fixtures/bench/*.json`)

| trace | requests | prompt tokens (median / max / total) | sha256 |
|---|---:|---|---|
| bench_main | 377 | 2,519 / 3,427 / 955,528 | `cee5445d…ef000` |
| bench_xsession | 155 | 2,419 / 2,840 / 374,776 | `271a8a6f…9df` |

## Predicted token-weighted hit rate (concurrency 1, block 16)

| trace | policy | client rendering | vLLM rendering |
|---|---|---:|---:|
| bench_main | raw | 73.76% | 74.63% |
| bench_main | L1 (session-stable tool order) | 81.77% | 81.82% |
| bench_xsession | raw | 86.44% | 86.48% |
| bench_xsession | L1 session-stable (Phase 2 policy) | 86.50% | 86.55% |
| bench_xsession | L1 stateless name sort (Phase 1 policy) | 95.11% | 95.15% |

Predicted recomputed prefill tokens, vLLM rendering: bench_main raw 242,440 → L1 171,996 (−29.1%).

## Real-tokenizer token change from L1 (bench_main, all 377 requests)

| rendering | tools key sort only | whole L1 |
|---|---:|---:|
| client | −0.91% | −0.94% |
| vLLM | −0.95% | −0.98% |

With the fixture tokenizer the key sort had *grown* prompts by ~+1.8%; with the real Qwen2.5 tokenizer it
shrinks them slightly. (Computed on CPU; this is a tokenizer measurement, not a GPU measurement.)

## What these predictions do not cover

Concurrency effects, decode-token caching, preemption, the real KV capacity, and whether vLLM's rendering
path matches `prefixanvil.render.vllm_normalize` byte for byte. The per-request `prompt_tokens`
comparison in `fidelity.json` checks the last point directly.
