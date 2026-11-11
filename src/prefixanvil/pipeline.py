"""`prefixanvil analyze` pipeline: ingest -> render -> replay (i)/(ii)/(iii) -> classify."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .agentload import load_labels
from .canonical import RULES, SessionContext, canonicalize, next_session_context
from .classify import FAULT_NAMES, ClassifyConfig, Classifier, Item, PairResult, classify_all, evaluate
from .ingest import IngestResult, load_jsonl
from .render import Renderer, RenderError, load_tokenizer
from .simulator import PrefixCacheSimulator, SimConfig, aligned, max_reusable

DIV_BUCKETS = [0, 16, 64, 256, 1024, 4096, 16384]


@dataclass
class AnalyzeOptions:
    tokenizer: str | None = None            # None -> bundled fixture tokenizer
    chat_template: str | None = None
    allow_download: bool = False
    block_size: int = 16
    num_blocks: int = 8192
    hash_model_name: bool = False
    prefill_tokens_per_s: float = 10000.0
    order: str = "file"
    labels: str | None = None
    session_aware_l1: bool = True            # Phase 2 Step 0; False = Phase-1 stateless sort
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)


def _bucket(v: int) -> str:
    for lo, hi in zip(DIV_BUCKETS, DIV_BUCKETS[1:]):
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f"[{DIV_BUCKETS[-1]},inf)"


def _cond_summary(sim: PrefixCacheSimulator, tps: float) -> dict[str, Any]:
    s = sim.stats
    return {
        "requests": s.requests,
        "prompt_tokens": s.prompt_tokens,
        "cached_tokens": s.cached_tokens,
        "prefill_tokens": s.prefill_tokens,
        "hit_rate_token_weighted": s.hit_rate,
        "evictions": s.evictions,
        "oversize_requests": s.oversize_requests,
        "prefill_gpu_seconds_est": s.prefill_tokens / tps,
    }


def _cause_breakdown(results: list[PairResult]) -> list[dict]:
    agg: dict[str, dict] = {}
    total_lost = sum(r.lost_tokens for r in results) or 1
    for r in results:
        if r.label == "none":
            continue
        a = agg.setdefault(r.label, {"label": r.label, "description": FAULT_NAMES.get(r.label, ""),
                                     "pairs": 0, "lost_tokens": 0, "by_scope": Counter(),
                                     "details": Counter(), "examples": []})
        a["pairs"] += 1
        a["lost_tokens"] += r.lost_tokens
        a["by_scope"][r.scope] += 1
        a["details"][r.detail] += 1
        if len(a["examples"]) < 3:
            a["examples"].append({"request_index": r.request_index, "request_id": r.request_id,
                                  "reference_index": r.reference_index, "element": r.element,
                                  "first_divergence_token": r.first_divergence_token,
                                  "lost_tokens": r.lost_tokens, "detail": r.detail})
    out = []
    for a in agg.values():
        a["lost_share"] = a["lost_tokens"] / total_lost
        a["by_scope"] = dict(a["by_scope"])
        a["details"] = dict(a["details"].most_common(5))
        out.append(a)
    out.sort(key=lambda a: (-a["lost_tokens"], a["label"]))
    return out


def _divergence_distribution(results: list[PairResult]) -> dict[str, Any]:
    tok = Counter()
    rel = Counter()
    for r in results:
        if r.label == "none" or r.first_divergence_token is None:
            continue
        tok[_bucket(r.first_divergence_token)] += 1
        frac = r.first_divergence_token / max(1, r.prompt_tokens)
        rel[f"{min(9, int(frac * 10)) * 10}-{min(9, int(frac * 10)) * 10 + 10}%"] += 1
    order = [_bucket(b) for b in DIV_BUCKETS]
    return {
        "by_token_position": {k: tok[k] for k in order if tok[k]},
        "by_relative_position": dict(sorted(rel.items(), key=lambda kv: int(kv[0].split("-")[0]))),
    }


def analyze(trace_path: str, opts: AnalyzeOptions | None = None) -> dict[str, Any]:
    opts = opts or AnalyzeOptions()
    opts.classify.block_size = opts.block_size
    ing: IngestResult = load_jsonl(trace_path, order=opts.order)
    tok = load_tokenizer(opts.tokenizer, opts.chat_template, opts.allow_download)
    renderer = Renderer(tok)
    info = tok.template_info()

    raw_items: list[Item] = []
    canon_items: list[Item] = []
    render_failures: Counter = Counter()
    l1_applied: Counter = Counter()
    l1_skipped: Counter = Counter()
    l1_touched = 0
    l1_log_sample: list[dict] = []
    session_ctx: dict[str, SessionContext] = {}
    for req in ing.requests:
        try:
            rr = renderer.render(req)
        except RenderError as e:
            render_failures[str(e).split(":")[0]] += 1
            continue
        ctx = None
        if opts.session_aware_l1 and req.session_id:
            ctx = session_ctx.get(req.session_id, SessionContext())
        creq, log = canonicalize(req, ctx)
        if ctx is not None:
            session_ctx[req.session_id] = next_session_context(creq)
        try:
            cr = renderer.render(creq)
        except RenderError:     # fail-open: fall back to the raw request
            creq, cr = req, rr
            l1_skipped["*:render_error_after_canonicalization"] += 1
        raw_items.append(Item(req, rr))
        canon_items.append(Item(creq, cr))
        for e in log.applied:
            l1_applied[e["rule"]] += 1
        for e in log.skipped:
            l1_skipped[f"{e['rule']}:{e['reason']}"] += 1
        if log.applied:
            l1_touched += 1
            if len(l1_log_sample) < 5:
                l1_log_sample.append({"request_index": req.index, "request_id": req.request_id,
                                      "applied": log.applied[:4]})

    sim_cfg = SimConfig(opts.block_size, opts.num_blocks, opts.hash_model_name)
    sim_i = PrefixCacheSimulator(sim_cfg)
    sim_ii = PrefixCacheSimulator(sim_cfg)
    for it, ct in zip(raw_items, canon_items):
        sim_i.process(it.rendered.token_ids, it.req.model, it.req.index)
        sim_ii.process(ct.rendered.token_ids, ct.req.model, ct.req.index)

    clf_raw = Classifier(renderer, opts.classify)
    clf_canon = Classifier(renderer, opts.classify)
    res_raw = classify_all(raw_items, clf_raw)
    res_canon = classify_all(canon_items, clf_canon)

    # (iii) reference upper bound (see README): ideal stable prefix, infinite cache
    B = opts.block_size
    ub_cached = 0
    ub_prompt = 0
    max_len_in_session: dict[str, int] = {}
    seen_models: set[str] = set()
    for k, ct in enumerate(canon_items):
        n = len(ct.rendered.token_ids)
        cap = max_reusable(n, B)
        ub = max(sim_i.results[k].cached_tokens, sim_ii.results[k].cached_tokens)
        sid = ct.req.session_id
        if sid and sid in max_len_in_session:
            ub = max(ub, aligned(min(max_len_in_session[sid], n - 1), B))
        elif ct.req.model in seen_models:
            ub = max(ub, aligned(min(clf_canon._static_tokens(ct), n - 1), B))
        ub = min(ub, cap)
        ub_cached += ub
        ub_prompt += n
        if sid:
            max_len_in_session[sid] = max(max_len_in_session.get(sid, 0), n)
        seen_models.add(ct.req.model)

    tps = opts.prefill_tokens_per_s
    c_i = _cond_summary(sim_i, tps)
    c_ii = _cond_summary(sim_ii, tps)
    c_iii = {
        "requests": len(canon_items), "prompt_tokens": ub_prompt, "cached_tokens": ub_cached,
        "prefill_tokens": ub_prompt - ub_cached,
        "hit_rate_token_weighted": ub_cached / ub_prompt if ub_prompt else 0.0,
        "prefill_gpu_seconds_est": (ub_prompt - ub_cached) / tps,
        "note": "UPPER BOUND, not an achievable target: every request assumed to extend its "
                "session's longest earlier prompt and to share the static prefix (system+tools) "
                "across sessions; infinite cache, no eviction.",
    }
    pf_i, pf_ii = c_i["prefill_tokens"], c_ii["prefill_tokens"]

    report: dict[str, Any] = {
        "prefixanvil_version": __version__,
        "trace": trace_path,
        "ingest": ing.summary() | {"render_failures": dict(render_failures),
                                   "replayed": len(raw_items)},
        "tokenizer": info,
        "simulator": sim_cfg.as_dict() | {
            "model": "spec-based approximation of vLLM V1 APC (not a vLLM reproduction)",
            "sequential_requests": True, "decode_tokens_simulated": False,
            "last_token_always_recomputed": True},
        "throughput_assumption": {"prefill_tokens_per_s": tps,
                                  "note": "assumed constant prefill throughput; GPU-seconds are a "
                                          "linear estimate, not a measurement"},
        "conditions": {"i_raw": c_i, "ii_l1": c_ii, "iii_upper_bound": c_iii},
        "savings_l1": {
            "prefill_tokens_saved": pf_i - pf_ii,
            "prefill_reduction_ratio": (pf_i - pf_ii) / pf_i if pf_i else 0.0,
            "gpu_seconds_saved_est": (pf_i - pf_ii) / tps,
        },
        "l1_transforms": {"rules": list(RULES), "requests_touched": l1_touched,
                          "tool_sort": "session_aware" if opts.session_aware_l1 else "stateless",
                          "applied_by_rule": dict(sorted(l1_applied.items())),
                          "skipped_fail_open": dict(sorted(l1_skipped.items())),
                          "sample": l1_log_sample},
        "pairs": {"total": len(res_raw),
                  "by_scope": dict(Counter(r.scope for r in res_raw)),
                  "fragmented_raw": sum(r.label != "none" for r in res_raw),
                  "fragmented_after_l1": sum(r.label != "none" for r in res_canon)},
        "first_divergence": _divergence_distribution(res_raw),
        "first_miss_block_raw": dict(sorted(Counter(
            _bucket(r.first_miss_block * B) for r in sim_i.results if r.first_miss_block is not None
        ).items(), key=lambda kv: int(kv[0][1:].split(",")[0]))),
        "causes_raw": _cause_breakdown(res_raw),
        "causes_residual_after_l1": _cause_breakdown(res_canon),
        "pair_results": [r.as_dict() for r in res_raw],
    }
    if opts.labels:
        labels = load_labels(opts.labels)
        report["classifier_evaluation"] = evaluate(res_raw, labels) | {"labels_file": opts.labels}
    return report
