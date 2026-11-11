# PrefixAnvil benchmark

**Hardware: Tesla T4 (15360 MiB), Tesla T4 (15360 MiB) — kaggle free tier.** TTFT and GPU-seconds below are **T4 reference values**, not representative of H100-class GPUs; an H100 measurement is future work. Hit rate and prefill-token counts are determined by the cache algorithm and do not depend on the GPU generation.

## Environment

- Platform: kaggle; GPUs: Tesla T4 (15360 MiB), Tesla T4 (15360 MiB); driver 580.178.04 / driver CUDA 13.0
- vLLM: `vllm==0.33.0` (installed 0.33.0, torch 2.14.0+cu132, CUDA 13.2)
- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ` @ `b25037543e9394b818fdfca67ab2a00ecc7dd641` (license apache-2.0)
- Attention backend: TRITON_ATTN; block size 16; KV blocks 7148; prefix caching True
- Server command: `/tmp/pfvenv/bin/vllm serve /tmp/pf_models/Qwen__Qwen2.5-7B-Instruct-AWQ --served-model-name pf-model --port 8000 --host 127.0.0.1 --enable-prefix-caching --enable-prompt-tokens-details --dtype half --max-model-len 8192 --gpu-memory-utilization 0.9 --seed 0 --enable-auto-tool-choice --tool-call-parser hermes`
- Trace `bench_main`: 377 requests, sha256 `cee5445df3364b3f…` (AgentLoad, seed fixed in `fixtures/bench/bench_main.json`)
- Trace `bench_xsession`: 155 requests, sha256 `271a8a6f3c2a61e9…` (AgentLoad, seed fixed in `fixtures/bench/bench_xsession.json`)
- Replay: temperature 0, max_tokens 64, open-loop (recorded assistant turns, not model outputs), session order preserved. Cache reset (verified by a probe) before every pass; warmup pass discarded.
- Budget decisions taken by the runner: none (no budget cuts)

## Headline: measured vs pre-registered

Pre-registered predictions (vLLM server-side rendering) were committed before any GPU run (`docs/bench-preregistration.md`, `fixtures/bench/preregistered_predictions.json`); they are not measurements.

| metric | measured | pre-registered (vLLM rendering) | Δ (pts) |
|---|---:|---:|---:|
| bench_main raw (hit rate) | 74.71% | 74.63% | +0.08 |
| bench_main L1 (hit rate) | 81.79% | 81.82% | -0.03 |
| bench_xsession session-pinned (hit rate) | 86.69% | 86.55% | +0.15 |
| bench_xsession stateless name-sort (hit rate) | 95.31% | 95.15% | +0.16 |

**Recomputed prefill tokens (bench_main, concurrency 1):** raw 241,640 → L1 172,268 (**-28.7%**) → L2 155,068 (**-35.8%**). Token-weighted hit rate 74.71% → 81.79% → 83.61%.

## Main result (concurrency 1)

| condition | passes | hit rate (token-weighted, per pass) | recomputed prefill tokens | TTFT p50 / p95 (T4, s) | T4 GPU-s per 1k req |
|---|---:|---|---|---|---:|
| (a) raw, direct to vLLM | 2 | 74.7% / 74.7% | 241,640 / 241,640 | 0.533 / 4.264 | 2,488 |
| (b) prefixanvil proxy L1 | 2 | 81.8% / 81.8% | 172,268 / 172,268 | 0.519 / 4.011 | 2,157 |
| (c) prefixanvil proxy L2 | 2 | 83.6% / 83.6% | 155,068 / 155,068 | 0.517 / 3.573 | 2,114 |

The L2 row (volatile-line relocation on top of L1) is a reference condition: it is the single L2 transform Phase 2 ships, shown here for the extra prefill it recovers.

## TTFT reference (concurrency 8, T4) — reference only

Concurrency-8 rows are a T4 reference for latency under load; they are not part of the pre-registered concurrency-1 table above.

| condition | hit rate | TTFT p50 / p95 (s) |
|---|---|---|
| TTFT reference, concurrency 8, raw | 74.7% | 1.056 / 5.661 |
| TTFT reference, concurrency 8, L1 | 82.0% | 0.764 / 5.045 |

## Cross-session tool order (what Phase 2's session-stable policy gives up)

| condition | hit rate | recomputed prefill tokens |
|---|---|---|
| cross-session: raw | 86.6% | 50,120 |
| cross-session: L1 session-stable tool order (Phase 2 policy) | 86.7% | 49,273 |
| cross-session: L1 stateless name sort (Phase 1 policy) | 95.3% | 17,369 |

**Disclosure — session-pinned vs stateless.** On bench_xsession, session-pinned ordering measured 86.69% and stateless name-sort measured 95.31%. Both are reported as measured. Session pinning is the conservative, deployable Phase 2 policy — it preserves each session's own tool order and never reorders across sessions — so the higher stateless number is **not** presented as the winner on this trace set; it is the ceiling a stateless global sort would reach by discarding session order.

## Simulator fidelity (vLLM-faithful rendering)

Simulator re-run with the real tokenizer, block size 16 and 7148 KV blocks, rendered the way the vLLM server normalizes requests (the deployable path).

| condition | sim hit | real hit | Δ (pts) | sim prefill | real prefill | Δ prefill | prompt tokens exact |
|---|---:|---:|---:|---:|---:|---:|---|
| main_a_raw | 74.6% | 74.7% | -0.08 | 242,440 | 241,640 | +0.3% | 377/377 |
| main_b_l1 | 81.8% | 81.8% | +0.03 | 171,996 | 172,268 | -0.2% | 377/377 |
| xs_raw | 86.5% | 86.6% | -0.15 | 50,680 | 50,120 | +1.1% | 155/155 |
| xs_session | 86.5% | 86.7% | -0.15 | 49,817 | 49,273 | +1.1% | 155/155 |
| xs_stateless | 95.2% | 95.3% | -0.16 | 17,945 | 17,369 | +3.3% | 155/155 |

Prompt-token totals were **exact per request** in this (vLLM) rendering: every condition shows prompt tokens exact = requests, so the simulator and the server tokenize each prompt identically; the residual hit-rate gap is block-boundary rounding only.

Prompt-token change from L1 with the real tokenizer (vLLM rendering): key sort -0.95%, whole L1 -0.98%.

## Real-model output equivalence sampling (greedy, max_tokens 64, cold cache per send)

| comparison | pairs | exact match | rate |
|---|---:|---:|---:|
| control_raw_vs_raw | 18 | 18 | 100.0% |
| warm_cache_vs_cold | 18 | 17 | 94.4% |
| L1_vs_raw | 18 | 10 | 55.6% |
| L1_vs_raw_changed_only | 14 | 6 | 42.9% |
| L2_vs_raw | 6 | 4 | 66.7% |
| L2_vs_raw_changed_only | 3 | 1 | 33.3% |
| sampling errors | | 1 | |

**Disclosure — read this before the hit-rate tables.** L1 and L2 are semantics-preserving *prompt* transforms (deterministic canonicalization, plus one volatile-line relocation for L2); they are **not** guaranteed to produce byte-identical generated text. This stage sampled real-model outputs on synthetic fixture prompts and compared them verbatim:

- Control raw-vs-raw: 18/18 identical — the sampler itself is deterministic.
- Warm-cache vs cold: 17/18 — **cache state alone flipped 1 of 18 outputs**, with no prompt change at all.
- L1 vs raw: 10/18 identical (changed-only 6/14); L2 vs raw: 4/6 identical (changed-only 1/3); 1 sampling error.

These are **sampled generation-match rates, not an equivalence guarantee**. Equivalence of the transforms is enforced separately by the deterministic suite in `fixtures/equiv/` (CI gate); the rates here show how often the real model's sampled text happened to match, against a raw-vs-raw control and a cache-only control. Mismatching pairs are kept verbatim in `equiv_sample.json` and classified by hand (meaning change vs formatting).

## NVIDIA Dynamo trial (2 × T4, time-boxed) — separate run

A separate trial, run on its own (90.0-minute time box). **Different traces and conditions from the Step 1 measurement above; not part of the pre-registered table.**

- Environment: `ai-dynamo[vllm]==1.6.2`, vLLM 0.31.0, torch 2.13.0+cu132 (CUDA 13.2), Tesla T4, Tesla T4, driver CUDA 13.0
- Status: **complete**

| condition | router | proxy L1 | requests | hit rate (worker counters) | errors | TTFT p50 / p95 (s) |
|---|---|---|---:|---:|---:|---|
| a_rr_raw | round-robin | False | 150 | 60.4% | 0 | 0.987 / 4.267 |
| b_kv_raw | kv | False | 150 | 60.2% | 0 | 0.781 / 4.243 |
| c_kv_l1 | kv | True | 150 | 65.2% | 0 | 0.625 / 4.231 |

Worker-summed hit rate across the two vLLM workers; KV-aware routing with L1 canonicalization recovers the most reuse of the three. TTFT is a T4 reference value.

## Limitations

- Single free-tier T4: TTFT/GPU-seconds are T4 reference values only. H100 numbers are future work.
- Synthetic agent workload (AgentLoad); no production logs.
- Concurrency-1 replay isolates cache accounting; it is not a throughput benchmark.
- L1/L2 are prompt transforms; generated text is not guaranteed identical (see the sampling section).
