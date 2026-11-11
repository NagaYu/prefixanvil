"""`prefixanvil gate`: CI regression gate on top of `prefixanvil analyze`.

Re-runs the analyzer on a trace and compares against a versioned baseline
(``prefixanvil.gate-baseline/v1``). Fails when, for condition (i) raw or
(ii) L1, the token-weighted hit rate drops by more than ``max_hitrate_drop``
(absolute, e.g. 0.01 = 1 percentage point) or recomputed prefill tokens grow by
more than ``max_prefill_growth`` (relative, e.g. 0.02 = 2%). The baseline is
only written with ``--write-baseline``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from . import __version__
from .pipeline import AnalyzeOptions, analyze

BASELINE_SCHEMA = "prefixanvil.gate-baseline/v1"
CONDITIONS = ("i_raw", "ii_l1")
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:24]


def analyzer_settings(report: dict, opts: AnalyzeOptions) -> dict:
    return {"block_size": opts.block_size, "num_blocks": opts.num_blocks,
            "hash_model_name": opts.hash_model_name, "session_aware_l1": opts.session_aware_l1,
            "order": opts.order, "tokenizer_is_fixture": report["tokenizer"]["tokenizer_is_fixture"],
            "template_version": report["tokenizer"]["template_version"]}


def metrics(report: dict) -> dict:
    out = {}
    for c in CONDITIONS:
        d = report["conditions"][c]
        out[c] = {"hit_rate": d["hit_rate_token_weighted"], "prefill_tokens": d["prefill_tokens"],
                  "prompt_tokens": d["prompt_tokens"], "requests": d["requests"]}
    return out


def build_baseline(trace: str, report: dict, opts: AnalyzeOptions) -> dict:
    return {"schema": BASELINE_SCHEMA, "prefixanvil_version": __version__,
            "trace": {"path": os.path.basename(trace), "sha256": _sha(trace)},
            "analyzer": analyzer_settings(report, opts), "metrics": metrics(report)}


@dataclass
class GateResult:
    passed: bool
    rows: list[dict]
    notes: list[str]
    exit_code: int


def compare(baseline: dict, current: dict, max_hitrate_drop: float, max_prefill_growth: float) -> GateResult:
    rows, ok = [], True
    for c in CONDITIONS:
        b, n = baseline["metrics"][c], current["metrics"][c]
        drop = b["hit_rate"] - n["hit_rate"]
        hit_ok = drop <= max_hitrate_drop + 1e-12
        rows.append({"metric": f"{c}.hit_rate", "baseline": b["hit_rate"], "current": n["hit_rate"],
                     "delta": n["hit_rate"] - b["hit_rate"], "limit": f"drop <= {max_hitrate_drop:.4f}", "ok": hit_ok})
        growth = (n["prefill_tokens"] - b["prefill_tokens"]) / b["prefill_tokens"] if b["prefill_tokens"] \
            else (0.0 if n["prefill_tokens"] == 0 else float("inf"))
        pf_ok = growth <= max_prefill_growth + 1e-12
        rows.append({"metric": f"{c}.prefill_tokens", "baseline": b["prefill_tokens"],
                     "current": n["prefill_tokens"], "delta": growth,
                     "limit": f"growth <= {max_prefill_growth:.2%}", "ok": pf_ok})
        ok = ok and hit_ok and pf_ok
    notes = []
    if baseline["trace"]["sha256"] != current["trace"]["sha256"]:
        notes.append("trace bytes differ from the baseline's trace (expected if the generator or config changed)")
    return GateResult(ok, rows, notes, EXIT_PASS if ok else EXIT_FAIL)


def format_table(rows: list[dict]) -> str:
    def fmt(metric: str, v: Any) -> str:
        return f"{v:.4f}" if metric.endswith("hit_rate") else f"{v:,}"
    lines = ["| metric | baseline | current | delta | limit | status |",
             "|---|---:|---:|---:|---|---|"]
    for r in rows:
        delta = f"{r['delta']:+.4f}" if r["metric"].endswith("hit_rate") else f"{r['delta']:+.2%}"
        lines.append(f"| {r['metric']} | {fmt(r['metric'], r['baseline'])} | {fmt(r['metric'], r['current'])} "
                     f"| {delta} | {r['limit']} | {'PASS' if r['ok'] else 'FAIL'} |")
    return "\n".join(lines)


def run_gate(trace: str, baseline_path: str, opts: AnalyzeOptions | None = None,
             max_hitrate_drop: float = 0.01, max_prefill_growth: float = 0.02,
             write_baseline: bool = False, out=print) -> int:
    opts = opts or AnalyzeOptions()
    if not os.path.exists(trace):
        out(f"prefixanvil gate: trace not found: {trace}")
        return EXIT_ERROR
    report = analyze(trace, opts)
    current = build_baseline(trace, report, opts)
    old = None
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("schema") != BASELINE_SCHEMA:
            out(f"prefixanvil gate: baseline schema {old.get('schema')!r} != {BASELINE_SCHEMA}")
            return EXIT_ERROR
    if write_baseline:
        if old is not None:
            out(format_table(compare(old, current, max_hitrate_drop, max_prefill_growth).rows))
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
            f.write("\n")
        out(f"prefixanvil gate: baseline written to {baseline_path}")
        return EXIT_PASS
    if old is None:
        out(f"prefixanvil gate: no baseline at {baseline_path} (create one explicitly with --write-baseline)")
        return EXIT_ERROR
    if old["analyzer"] != current["analyzer"]:
        diff = {k: (old["analyzer"].get(k), current["analyzer"].get(k)) for k in
                set(old["analyzer"]) | set(current["analyzer"]) if old["analyzer"].get(k) != current["analyzer"].get(k)}
        out(f"prefixanvil gate: analyzer settings differ from baseline, not comparable: {diff}")
        return EXIT_ERROR
    res = compare(old, current, max_hitrate_drop, max_prefill_growth)
    out(format_table(res.rows))
    for n in res.notes:
        out(f"note: {n}")
    out(f"prefixanvil gate: {'PASS' if res.passed else 'FAIL'}")
    return res.exit_code
