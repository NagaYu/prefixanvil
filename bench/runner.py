"""Condition runner: warmup -> verified cache reset -> measured pass(es), saved per pass.

Cache protocol per condition (documented in RUNBOOK/BENCH):
  reset+verify -> warmup pass (discarded; warms kernels/graphs, NOT the cache)
  -> reset+verify -> measured pass 1 -> reset+verify -> measured pass 2 (if budget allows)
A warmup that left its KV blocks in the cache would make the measured pass hit
everything, so the cache is always reset between passes.

Outputs ``<out_dir>/<condition_id>.json`` (written atomically after every pass)
and ``<out_dir>/plan_state.json``. Completed conditions are skipped on re-run
(resume); a partially completed condition is restarted from scratch because the
server-side cache state cannot be resumed.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import BENCH_SCHEMA_VERSION
from .metrics import diff, snapshot
from .replay import build_items, load_trace, replay, summarize
from .vllm_ctl import metrics_text, reset_prefix_cache, verify_reset

CONDITION_SCHEMA = "prefixanvil.bench.condition/v1"


@dataclass
class Condition:
    id: str
    trace: str                       # path to JSONL
    target: str = "direct"           # direct | proxy
    proxy_mode: str | None = None    # L1 | L2 (target=proxy)
    tool_order: str = "session"      # proxy PREFIXANVIL_L1_TOOL_ORDER
    concurrency: int = 1
    passes: int = 2
    warmup: str | int = "full"       # "full" | "none" | N first requests
    optional: bool = False
    label: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown condition keys {sorted(bad)}")
        c = cls(**d)
        if c.target not in ("direct", "proxy") or (c.target == "proxy" and c.proxy_mode not in ("L1", "L2")):
            raise ValueError(f"condition {c.id}: bad target/proxy_mode")
        return c


@dataclass
class RunConfig:
    base_url: str
    served_model: str
    out_dir: str
    max_tokens: int = 64
    budget_s: float | None = None     # wall-clock budget for the whole plan
    limit: int | None = None          # smoke runs: first N requests of every trace
    versions: dict = field(default_factory=dict)


ProxyFactory = Callable[[Condition], "contextlib.AbstractContextManager[str]"]


@contextlib.contextmanager
def _direct(cond: Condition) -> Iterator[str]:
    yield ""


def _atomic_write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _status(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("status")
    except (OSError, json.JSONDecodeError):
        return None


class Runner:
    def __init__(self, cfg: RunConfig, proxy_factory: ProxyFactory | None = None,
                 restart_server: Callable[[], None] | None = None, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.proxy_factory = proxy_factory
        self.restart_server = restart_server
        self.log = log
        self.t_start = time.time()
        self.decisions: list[str] = []
        self.s_per_request: float | None = None
        os.makedirs(cfg.out_dir, exist_ok=True)

    # -- cache reset -------------------------------------------------------
    def reset(self) -> dict:
        r = reset_prefix_cache(self.cfg.base_url)
        if r["status"] == "unsupported":
            if not self.restart_server:
                raise RuntimeError("prefix cache reset endpoint unavailable and no restart_server callback")
            self.restart_server()
            r = {"status": "restarted"}
        v = verify_reset(self.cfg.base_url, self.cfg.served_model) if r["status"] == "ok" else None
        if v is not None and not v["verified"]:
            raise RuntimeError(f"prefix cache reset could not be verified: {v}")
        return {"reset": r, "verify": v}

    # -- budget --------------------------------------------------------------
    def _remaining(self) -> float | None:
        if self.cfg.budget_s is None:
            return None
        return self.cfg.budget_s - (time.time() - self.t_start)

    def _estimate(self, n_requests: int, passes: int) -> float | None:
        return None if self.s_per_request is None else self.s_per_request * n_requests * passes

    # -- one pass --------------------------------------------------------------
    def _pass(self, cond: Condition, url: str, items, kind: str, n: int) -> dict:
        self.log(f"[{cond.id}] {kind} pass {n}: {len(items)} requests, concurrency {cond.concurrency}")
        before_txt = metrics_text(self.cfg.base_url)
        t0 = time.time()
        results = replay(items, url, concurrency=cond.concurrency)
        wall = time.time() - t0
        after_txt = metrics_text(self.cfg.base_url)
        sb, sa = snapshot(before_txt), snapshot(after_txt)
        if cond.concurrency <= 1 and items:
            self.s_per_request = wall / len(items)
        rec = {"kind": kind, "n": n, "wall_s": wall,
               "counters": diff(sb, sa), "metric_names": sa["names"], "missing_metrics": sa["missing"],
               "summary": summarize(results, wall).__dict__,
               "results": [r.as_dict() for r in results]}
        if sa["missing"]:
            rec["raw_metrics_after"] = after_txt     # names drifted: keep the evidence
        return rec

    # -- one condition -----------------------------------------------------------
    def run_condition(self, cond: Condition) -> dict:
        path = os.path.join(self.cfg.out_dir, f"{cond.id}.json")
        if _status(path) == "complete":
            self.log(f"[{cond.id}] already complete, skipping (resume)")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        lines = load_trace(cond.trace)
        items = build_items(lines, self.cfg.served_model, self.cfg.max_tokens, self.cfg.limit)
        passes, warmup = cond.passes, cond.warmup
        rem = self._remaining()
        est_one = self._estimate(len(items), 1)
        if rem is not None and est_one is not None:
            need = est_one * (passes + (1 if warmup == "full" else 0))
            if need > rem and passes > 1:
                passes = 1
                self.decisions.append(f"{cond.id}: measured passes 2 -> 1 (budget: need~{need:.0f}s, left {rem:.0f}s)")
            need = est_one * (passes + (1 if warmup == "full" else 0))
            if need > rem and warmup == "full":
                warmup = 30
                self.decisions.append(f"{cond.id}: warmup full -> first 30 requests (budget)")
            if est_one * passes > rem:
                if cond.optional:
                    self.decisions.append(f"{cond.id}: skipped (optional, budget)")
                    rec = {"schema": CONDITION_SCHEMA, "condition": cond.__dict__, "status": "skipped_budget"}
                    _atomic_write(path, rec)
                    return rec
                self.decisions.append(f"{cond.id}: may exceed budget (required condition, running anyway)")

        rec = {"schema": CONDITION_SCHEMA, "bench_schema": BENCH_SCHEMA_VERSION, "condition": cond.__dict__,
               "effective": {"passes": passes, "warmup": warmup, "requests": len(items),
                             "max_tokens": self.cfg.max_tokens, "limit": self.cfg.limit},
               "versions": self.cfg.versions, "started_at": time.time(), "status": "running", "passes": []}
        _atomic_write(path, rec)
        factory = self.proxy_factory if cond.target == "proxy" else _direct
        if factory is None:
            raise RuntimeError("proxy condition but no proxy_factory")
        with factory(cond) as proxy_url:
            url = proxy_url or self.cfg.base_url
            if warmup != "none":
                rec["reset_before_warmup"] = self.reset()
                w_items = items if warmup == "full" else items[: int(warmup)]
                w = self._pass(cond, url, w_items, "warmup", 0)
                w.pop("results")                 # discarded by design; keep only its summary
                rec["warmup"] = w
                _atomic_write(path, rec)
            for n in range(1, passes + 1):
                reset = self.reset()
                p = self._pass(cond, url, items, "measure", n)
                p["reset"] = reset
                rec["passes"].append(p)
                _atomic_write(path, rec)
        rec["status"] = "complete"
        rec["finished_at"] = time.time()
        _atomic_write(path, rec)
        return rec

    def run_plan(self, conditions: list[Condition]) -> dict:
        state_path = os.path.join(self.cfg.out_dir, "plan_state.json")
        out = {"conditions": {}, "decisions": self.decisions}
        for c in conditions:
            try:
                r = self.run_condition(c)
                out["conditions"][c.id] = r.get("status")
            except Exception as e:
                out["conditions"][c.id] = f"error:{type(e).__name__}: {e}"
                self.log(f"[{c.id}] ERROR {type(e).__name__}: {e}")
                if not c.optional:
                    _atomic_write(state_path, out)
                    raise
            _atomic_write(state_path, out)
        return out


def load_plan(path: str, trace_dir: str) -> list[Condition]:
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    conds = []
    for d in plan["conditions"]:
        d = dict(d)
        if not os.path.isabs(d["trace"]):
            d["trace"] = os.path.join(trace_dir, d["trace"])
        conds.append(Condition.from_dict(d))
    return conds
