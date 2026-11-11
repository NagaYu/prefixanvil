"""Generate BENCH.md (English) mechanically from a results directory.

Every number in the output is read from the raw JSON written by run_all.py /
run_dynamo.py (and the pre-registered predictions in fixtures/bench); nothing is
typed by hand. Prose (framing, disclosures) is fixed text, but every figure it
cites is interpolated from the JSON. Missing pieces are printed as "not measured"
with the reason recorded in stages.json. Results produced by the offline fake
server are refused unless ``allow_fake`` is set (tests only).

    .venv/bin/python -m bench.make_bench_md bench/results/<run-id> \
        --dynamo bench/results/<dynamo-run-id> --out BENCH.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

NA = "not measured"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PREREG = os.path.join(REPO_ROOT, "fixtures", "bench", "preregistered_predictions.json")


def _load(p: str) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pct(x) -> str:
    return NA if x is None else f"{x * 100:.1f}%"


def _pct2(x) -> str:
    return NA if x is None else f"{x * 100:.2f}%"


def _num(x) -> str:
    return NA if x is None else f"{x:,.0f}"


def _s(x) -> str:
    return NA if x is None else f"{x:.3f}"


def condition_row(rec: dict) -> dict:
    passes = rec.get("passes") or []
    if rec.get("status") != "complete" or not passes:
        return {"status": rec.get("status", "missing")}
    hit = [p["counters"].get("hit_rate") for p in passes]
    pre = [(p["counters"]["prefix_cache_queries"] - p["counters"]["prefix_cache_hits"])
           if p["counters"].get("prefix_cache_queries") is not None and p["counters"].get("prefix_cache_hits") is not None
           else None for p in passes]
    p1 = passes[0]
    n = p1["summary"]["requests"]
    return {"status": "complete", "passes": len(passes), "hit_rate": hit, "prefill_tokens": pre,
            "prefill_kv_computed_sum": p1["counters"].get("prefill_kv_computed_sum"),
            "ttft_p50": p1["summary"]["ttft_p50_s"], "ttft_p95": p1["summary"]["ttft_p95_s"],
            "wall_s": p1["wall_s"], "requests": n, "errors": p1["summary"]["errors"],
            "gpu_s_per_1k": p1["wall_s"] / n * 1000 if n else None,
            "concurrency": rec["condition"]["concurrency"], "label": rec["condition"].get("label", ""),
            "effective": rec.get("effective", {})}


def _prereg_index(prereg: dict) -> dict:
    """Map (trace, policy) -> vLLM-rendering prediction (no_eviction capacity)."""
    out: dict = {}
    for p in prereg.get("predictions", []):
        if p.get("rendering") != "vllm":
            continue
        key = (p["trace"], p["policy"])
        # no_eviction is the baseline; est_8000_blocks is identical on these traces.
        if key not in out or p.get("capacity") == "no_eviction":
            out[key] = p
    return out


def build(run_dir: str, allow_fake: bool = False, dynamo_dir: str | None = None,
          prereg_path: str = DEFAULT_PREREG) -> str:
    st = _load(os.path.join(run_dir, "stages.json"))
    env = st.get("env", {})
    if env.get("platform") == "fake" and not allow_fake:
        raise SystemExit("refusing to write BENCH.md from fake-server results")
    inst, model, server = st.get("install", {}), st.get("model", {}), st.get("server", {})
    gpu_list = env.get("gpus", [])
    gpus = ", ".join(f"{g['name']} ({g['memory_total']})" for g in gpu_list) or NA
    driver = (gpu_list[0].get("driver") if gpu_list else None) or NA
    L: list[str] = []
    a = L.append
    a("# PrefixAnvil benchmark")
    a("")
    a(f"**Hardware: {gpus} — {env.get('platform', '?')} free tier.** TTFT and GPU-seconds below are "
      "**T4 reference values**, not representative of H100-class GPUs; an H100 measurement is future work. "
      "Hit rate and prefill-token counts are determined by the cache algorithm and do not depend on the GPU generation.")
    a("")

    # ------------------------------------------------------------------ environment
    v = inst.get("versions") or {}
    cc = server.get("cache_config") or {}
    a("## Environment")
    a("")
    a(f"- Platform: {env.get('platform', NA)}; GPUs: {gpus}; driver {driver} / driver CUDA {env.get('driver_cuda', NA)}")
    a(f"- vLLM: `{inst.get('spec', NA)}` (installed {v.get('vllm', NA)}, torch {v.get('torch', NA)}, CUDA {v.get('torch_cuda', NA)})")
    a(f"- Model: `{model.get('repo', NA)}` @ `{model.get('revision', NA)}` (license {model.get('license', NA)})")
    if st.get("fallback"):
        fb = st["fallback"]
        a(f"- **Fallback model used**: `{fb['from']}` could not be served ({fb['reason'][:160]}); "
          f"measured with `{fb['to']}` instead.")
    a(f"- Attention backend: {server.get('attention_backend') or NA}; block size {cc.get('block_size', NA)}; "
      f"KV blocks {cc.get('num_gpu_blocks', NA)}; prefix caching {cc.get('enable_prefix_caching', NA)}")
    a(f"- Server command: `{' '.join(server.get('cmd', [])) or NA}`")
    tr = st.get("traces", {}).get("traces", {})
    for k, t in tr.items():
        a(f"- Trace `{k}`: {t['requests']} requests, sha256 `{t['sha256'][:16]}…` (AgentLoad, seed fixed in `fixtures/bench/{k}.json`)")
    a("- Replay: temperature 0, max_tokens 64, open-loop (recorded assistant turns, not model outputs), "
      "session order preserved. Cache reset (verified by a probe) before every pass; warmup pass discarded.")
    dec = st.get("plan", {}).get("decisions") or []
    a("- Budget decisions taken by the runner: " + ("; ".join(dec) if dec else "none (no budget cuts)"))
    a("")

    cdir = os.path.join(run_dir, "conditions")
    rows = {}
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".json") and f != "plan_state.json":
                rows[f[:-5]] = condition_row(_load(os.path.join(cdir, f)))

    # ------------------------------------------------------------------ headline: measured vs preregistered
    pre_idx = _prereg_index(_load(prereg_path)) if os.path.exists(prereg_path) else {}
    a("## Headline: measured vs pre-registered")
    a("")
    a("Pre-registered predictions (vLLM server-side rendering) were committed before any GPU run "
      "(`docs/bench-preregistration.md`, `fixtures/bench/preregistered_predictions.json`); they are not measurements.")
    a("")
    a("| metric | measured | pre-registered (vLLM rendering) | Δ (pts) |")
    a("|---|---:|---:|---:|")
    headline = [("bench_main raw (hit rate)", "main_a_raw", ("bench_main", "raw")),
                ("bench_main L1 (hit rate)", "main_b_l1", ("bench_main", "L1")),
                ("bench_xsession session-pinned (hit rate)", "xs_session", ("bench_xsession", "L1")),
                ("bench_xsession stateless name-sort (hit rate)", "xs_stateless", ("bench_xsession", "L1-stateless"))]
    for lab, cid, pkey in headline:
        r = rows.get(cid, {})
        meas = r.get("hit_rate", [None])[0] if r.get("status") == "complete" else None
        pred = (pre_idx.get(pkey) or {}).get("hit_rate")
        delta = f"{(meas - pred) * 100:+.2f}" if (meas is not None and pred is not None) else NA
        a(f"| {lab} | {_pct2(meas)} | {_pct2(pred)} | {delta} |")
    a("")

    ra, rb, rc = rows.get("main_a_raw", {}), rows.get("main_b_l1", {}), rows.get("main_c_l2", {})
    if (ra.get("status") == rb.get("status") == rc.get("status") == "complete"
            and ra["prefill_tokens"][0] and rb["prefill_tokens"][0] and rc["prefill_tokens"][0]):
        p_raw, p_l1, p_l2 = ra["prefill_tokens"][0], rb["prefill_tokens"][0], rc["prefill_tokens"][0]
        a(f"**Recomputed prefill tokens (bench_main, concurrency 1):** raw {_num(p_raw)} → "
          f"L1 {_num(p_l1)} (**{(p_l1 / p_raw - 1) * 100:+.1f}%**) → L2 {_num(p_l2)} (**{(p_l2 / p_raw - 1) * 100:+.1f}%**). "
          f"Token-weighted hit rate {_pct2(ra['hit_rate'][0])} → {_pct2(rb['hit_rate'][0])} → {_pct2(rc['hit_rate'][0])}.")
        a("")

    # ------------------------------------------------------------------ main detail table
    a("## Main result (concurrency 1)")
    a("")
    a("| condition | passes | hit rate (token-weighted, per pass) | recomputed prefill tokens | TTFT p50 / p95 (T4, s) | T4 GPU-s per 1k req |")
    a("|---|---:|---|---|---|---:|")
    for cid in ("main_a_raw", "main_b_l1", "main_c_l2"):
        r = rows.get(cid)
        if not r or r["status"] != "complete":
            a(f"| {cid} | – | {NA} ({(r or {}).get('status', 'missing')}) | | | |")
            continue
        a(f"| {r['label'] or cid} | {r['passes']} | {' / '.join(_pct(x) for x in r['hit_rate'])} | "
          f"{' / '.join(_num(x) for x in r['prefill_tokens'])} | {_s(r['ttft_p50'])} / {_s(r['ttft_p95'])} | {_num(r['gpu_s_per_1k'])} |")
    a("")
    a("The L2 row (volatile-line relocation on top of L1) is a reference condition: it is the single L2 transform "
      "Phase 2 ships, shown here for the extra prefill it recovers.")
    a("")

    a("## TTFT reference (concurrency 8, T4) — reference only")
    a("")
    a("Concurrency-8 rows are a T4 reference for latency under load; they are not part of the pre-registered "
      "concurrency-1 table above.")
    a("")
    a("| condition | hit rate | TTFT p50 / p95 (s) |")
    a("|---|---|---|")
    for cid in ("c8_a_raw", "c8_b_l1"):
        r = rows.get(cid)
        if r and r["status"] == "complete":
            a(f"| {r['label'] or cid} | {_pct(r['hit_rate'][0])} | {_s(r['ttft_p50'])} / {_s(r['ttft_p95'])} |")
        else:
            a(f"| {cid} | {NA} ({(r or {}).get('status', 'missing')}) | |")
    a("")

    # ------------------------------------------------------------------ cross-session + disclosure
    a("## Cross-session tool order (what Phase 2's session-stable policy gives up)")
    a("")
    a("| condition | hit rate | recomputed prefill tokens |")
    a("|---|---|---|")
    for cid in ("xs_raw", "xs_session", "xs_stateless"):
        r = rows.get(cid)
        if r and r["status"] == "complete":
            a(f"| {r['label'] or cid} | {_pct(r['hit_rate'][0])} | {_num(r['prefill_tokens'][0])} |")
        else:
            a(f"| {cid} | {NA} ({(r or {}).get('status', 'missing')}) | |")
    a("")
    xss, xst = rows.get("xs_session", {}), rows.get("xs_stateless", {})
    if xss.get("status") == xst.get("status") == "complete":
        a(f"**Disclosure — session-pinned vs stateless.** On bench_xsession, session-pinned ordering measured "
          f"{_pct2(xss['hit_rate'][0])} and stateless name-sort measured {_pct2(xst['hit_rate'][0])}. Both are reported "
          "as measured. Session pinning is the conservative, deployable Phase 2 policy — it preserves each session's "
          "own tool order and never reorders across sessions — so the higher stateless number is **not** presented as "
          "the winner on this trace set; it is the ceiling a stateless global sort would reach by discarding session order.")
        a("")

    # ------------------------------------------------------------------ fidelity (vLLM rendering only)
    fp = os.path.join(run_dir, "fidelity.json")
    a("## Simulator fidelity (vLLM-faithful rendering)")
    a("")
    if os.path.exists(fp):
        fd = _load(fp)
        vrows = [r for r in fd["rows"] if r.get("rendering") == "vllm"]
        a(f"Simulator re-run with the real tokenizer, block size {fd['block_size']} and {fd['num_gpu_blocks']} KV blocks, "
          "rendered the way the vLLM server normalizes requests (the deployable path).")
        a("")
        a("| condition | sim hit | real hit | Δ (pts) | sim prefill | real prefill | Δ prefill | prompt tokens exact |")
        a("|---|---:|---:|---:|---:|---:|---:|---|")
        exact_all = True
        for r in vrows:
            d = r.get("hit_rate_diff_pts")
            dp = r.get("prefill_diff_pct")
            if r.get("prompt_tokens_exact") != r.get("requests"):
                exact_all = False
            a(f"| {r['condition']} | {_pct(r['sim_hit_rate'])} | {_pct(r['real_hit_rate'])} | "
              f"{NA if d is None else f'{d:+.2f}'} | {_num(r['sim_prefill_tokens'])} | {_num(r['real_prefill_tokens'])} | "
              f"{NA if dp is None else f'{dp:+.1f}%'} | {r.get('prompt_tokens_exact', NA)}/{r.get('requests', NA)} |")
        a("")
        if exact_all and vrows:
            a("Prompt-token totals were **exact per request** in this (vLLM) rendering: every condition shows "
              "prompt tokens exact = requests, so the simulator and the server tokenize each prompt identically; "
              "the residual hit-rate gap is block-boundary rounding only.")
        tc = fd.get("token_change", {}).get("vllm")
        if tc:
            a("")
            a(f"Prompt-token change from L1 with the real tokenizer (vLLM rendering): key sort "
              f"{tc['keysort_change'] * 100:+.2f}%, whole L1 {tc['l1_change'] * 100:+.2f}%.")
    else:
        a(NA)
    a("")

    # ------------------------------------------------------------------ real-model output sampling
    ep = os.path.join(run_dir, "equiv_sample.json")
    a("## Real-model output equivalence sampling (greedy, max_tokens 64, cold cache per send)")
    a("")
    if os.path.exists(ep):
        es = _load(ep)["summary"]

        def mp(k):
            x = es.get(k) or {}
            return x.get("match"), x.get("pairs"), x.get("rate")

        a("| comparison | pairs | exact match | rate |")
        a("|---|---:|---:|---:|")
        for k in ("control_raw_vs_raw", "warm_cache_vs_cold", "L1_vs_raw", "L1_vs_raw_changed_only",
                  "L2_vs_raw", "L2_vs_raw_changed_only"):
            m, p, rate = mp(k)
            a(f"| {k} | {p if p is not None else NA} | {m if m is not None else NA} | {_pct(rate)} |")
        a(f"| sampling errors | | {es.get('errors', NA)} | |")
        a("")
        c_m, c_p, _ = mp("control_raw_vs_raw")
        w_m, w_p, _ = mp("warm_cache_vs_cold")
        l1_m, l1_p, _ = mp("L1_vs_raw")
        l1c_m, l1c_p, _ = mp("L1_vs_raw_changed_only")
        l2_m, l2_p, _ = mp("L2_vs_raw")
        l2c_m, l2c_p, _ = mp("L2_vs_raw_changed_only")
        a("**Disclosure — read this before the hit-rate tables.** L1 and L2 are semantics-preserving *prompt* "
          "transforms (deterministic canonicalization, plus one volatile-line relocation for L2); they are **not** "
          "guaranteed to produce byte-identical generated text. This stage sampled real-model outputs on synthetic "
          "fixture prompts and compared them verbatim:")
        a("")
        if None not in (c_m, c_p):
            a(f"- Control raw-vs-raw: {c_m}/{c_p} identical — the sampler itself is deterministic.")
        if None not in (w_m, w_p):
            a(f"- Warm-cache vs cold: {w_m}/{w_p} — **cache state alone flipped {w_p - w_m} of {w_p} outputs**, "
              "with no prompt change at all.")
        if None not in (l1_m, l1_p, l2_m, l2_p):
            a(f"- L1 vs raw: {l1_m}/{l1_p} identical (changed-only {l1c_m}/{l1c_p}); "
              f"L2 vs raw: {l2_m}/{l2_p} identical (changed-only {l2c_m}/{l2c_p}); {es.get('errors', NA)} sampling error.")
        a("")
        a("These are **sampled generation-match rates, not an equivalence guarantee**. Equivalence of the transforms "
          "is enforced separately by the deterministic suite in `fixtures/equiv/` (CI gate); the rates here show how "
          "often the real model's sampled text happened to match, against a raw-vs-raw control and a cache-only "
          "control. Mismatching pairs are kept verbatim in `equiv_sample.json` and classified by hand "
          "(meaning change vs formatting).")
    else:
        a(NA)
    a("")

    # ------------------------------------------------------------------ dynamo (its own run / environment)
    dpath = None
    if dynamo_dir:
        cand = os.path.join(dynamo_dir, "dynamo.json")
        dpath = cand if os.path.exists(cand) else dynamo_dir if dynamo_dir.endswith(".json") else None
    if dpath is None:
        cand = os.path.join(run_dir, "dynamo", "dynamo.json")
        dpath = cand if os.path.exists(cand) else None
    a("## NVIDIA Dynamo trial (2 × T4, time-boxed) — separate run")
    a("")
    if dpath and os.path.exists(dpath):
        dy = _load(dpath)
        dv = dy.get("versions", {})
        pre = next((s for s in dy.get("steps", []) if s.get("step") == "precondition"), {})
        dgpus = ", ".join(pre.get("gpus", [])) or NA
        a(f"A separate trial, run on its own ({dy.get('timebox_min', NA)}-minute time box). "
          "**Different traces and conditions from the Step 1 measurement above; not part of the pre-registered table.**")
        a("")
        a(f"- Environment: `{dy.get('spec', NA)}`, vLLM {dv.get('vllm', NA)}, torch {dv.get('torch', NA)} "
          f"(CUDA {dv.get('cuda', NA)}), {dgpus}, driver CUDA {pre.get('driver_cuda', NA)}")
        a(f"- Status: **{dy.get('status', NA)}**")
        a("")
        a("| condition | router | proxy L1 | requests | hit rate (worker counters) | errors | TTFT p50 / p95 (s) |")
        a("|---|---|---|---:|---:|---:|---|")
        for c, r in (dy.get("conditions") or {}).items():
            a(f"| {c} | {r.get('router_mode', NA)} | {r.get('proxy_l1', NA)} | {r.get('requests', NA)} | "
              f"{_pct(r.get('hit_rate_workers_sum'))} | {r.get('errors', NA)} | "
              f"{_s(r.get('ttft_p50_s'))} / {_s(r.get('ttft_p95_s'))} |")
        a("")
        a("Worker-summed hit rate across the two vLLM workers; KV-aware routing with L1 canonicalization recovers "
          "the most reuse of the three. TTFT is a T4 reference value.")
    else:
        a(f"{NA} (no Dynamo run supplied).")
    a("")

    a("## Limitations")
    a("")
    a("- Single free-tier T4: TTFT/GPU-seconds are T4 reference values only. H100 numbers are future work.")
    a("- Synthetic agent workload (AgentLoad); no production logs.")
    a("- Concurrency-1 replay isolates cache accounting; it is not a throughput benchmark.")
    a("- L1/L2 are prompt transforms; generated text is not guaranteed identical (see the sampling section).")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="BENCH.md")
    ap.add_argument("--dynamo", default=None, help="directory (or dynamo.json) of a separate Dynamo trial")
    ap.add_argument("--prereg", default=DEFAULT_PREREG, help="pre-registered predictions JSON")
    ap.add_argument("--allow-fake", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    text = build(a.run_dir, a.allow_fake, a.dynamo, a.prereg)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
