#!/usr/bin/env python3
"""PrefixAnvil Phase 3 Step 2 — NVIDIA Dynamo 2-worker experiment (best effort, time-boxed).

Run only after Step 1 has packaged its results. Requires 2 GPUs (Kaggle "GPU T4 x2").
Hard time box (default 90 min of wall clock for the whole script); when it runs out,
every process is stopped and the exact step that was in progress is recorded.

No Docker / etcd / NATS: Dynamo 1.6.2 supports ``--discovery-backend file``
(``DYN_FILE_KV``), a TCP request plane and a ZMQ event plane by default
(verified in the ai-dynamo 1.6.2 wheel: frontend/frontend_args.py,
common/configuration/groups/runtime_args.py). ``ai-dynamo[vllm]==1.6.2`` pins
``vllm==0.31.0`` and ``nixl[cu13]`` — i.e. it needs a CUDA 13 capable driver and
installs a *different* vLLM than Step 1 (recorded).

Conditions (all with the same reduced trace prefix, concurrency 1):
  (a) round-robin router, raw   (b) KV router, raw   (c) KV router, prefixanvil proxy L1 in front
Workers are restarted between conditions so every condition starts cold.
Results: results/<run-id>/dynamo/dynamo.json + raw metrics texts + logs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

SERVED = "pf-model"
FRONT_PORT = 8200
DYNAMO_SPEC = "ai-dynamo[vllm]==1.6.2"


class Box:
    def __init__(self, minutes: float):
        self.deadline = time.time() + minutes * 60

    def left(self) -> float:
        return self.deadline - time.time()

    def check(self, step: str) -> None:
        if self.left() <= 0:
            raise TimeoutError(f"time box exhausted before/while: {step}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--timebox-min", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=150, help="requests of bench_main per condition")
    ap.add_argument("--venv", default="/tmp/dynvenv")
    a = ap.parse_args()
    box = Box(a.timebox_min)
    run_dir = os.path.join(os.path.abspath(a.workdir), "results", a.run_id)
    out = os.path.join(run_dir, "dynamo")
    os.makedirs(out, exist_ok=True)
    rec: dict = {"schema": "prefixanvil.bench.dynamo/v1", "timebox_min": a.timebox_min, "steps": [],
                 "conditions": {}, "status": "running", "spec": DYNAMO_SPEC}
    procs: list[subprocess.Popen] = []

    def save() -> None:
        json.dump(rec, open(os.path.join(out, "dynamo.json"), "w"), indent=1)

    def step(name: str, **info) -> None:
        rec["steps"].append({"step": name, "t_min": round((time.time() - (box.deadline - a.timebox_min * 60)) / 60, 1),
                             **info})
        print(f"[dynamo] {name} {info}", flush=True)
        save()

    def stop_all() -> None:
        for p in procs:
            with contextlib.suppress(Exception):
                os.killpg(p.pid, signal.SIGTERM)
        for p in procs:
            with contextlib.suppress(Exception):
                p.wait(30)
        procs.clear()

    try:
        stages_p = os.path.join(run_dir, "stages.json")
        stages = json.load(open(stages_p)) if os.path.exists(stages_p) else {}
        if stages.get("package", {}).get("status") != "ok":
            rec["status"] = "not_run"
            step("precondition", ok=False, reason="Step 1 results are not packaged yet; run run_all.py first")
            return 2
        gpus = stages.get("env", {}).get("gpus", [])
        if len(gpus) < 2:
            rec["status"] = "not_run"
            step("precondition", ok=False, reason=f"needs 2 GPUs, found {len(gpus)}")
            return 0
        model_dir = stages["model"]["local_dir"]
        step("precondition", ok=True, gpus=[g["name"] for g in gpus], driver_cuda=stages["env"].get("driver_cuda"))

        # 1) install --------------------------------------------------------------
        box.check("install")
        log = os.path.join(out, "install.log")
        uv = [sys.executable, "-m", "uv"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        vpy = os.path.join(a.venv, "bin", "python")
        if not os.path.exists(vpy):
            subprocess.run(uv + ["venv", a.venv, "--python", "3.12", "--seed"], check=True,
                           stdout=open(log, "a"), stderr=subprocess.STDOUT)
        t0 = time.time()
        p = subprocess.run(uv + ["pip", "install", "--python", vpy, DYNAMO_SPEC, "--torch-backend=auto"],
                           stdout=open(log, "a"), stderr=subprocess.STDOUT, timeout=max(60, min(1800, box.left())))
        step("install", ok=p.returncode == 0, seconds=round(time.time() - t0))
        if p.returncode != 0:
            raise RuntimeError("ai-dynamo install failed (see dynamo/install.log)")
        p = subprocess.run(uv + ["pip", "install", "--python", vpy, "-e", a.src], stdout=open(log, "a"),
                           stderr=subprocess.STDOUT)
        probe = subprocess.run([vpy, "-c", "import json,torch,vllm,dynamo;print(json.dumps({'torch':torch.__version__,"
                                "'cuda':torch.version.cuda,'gpus':torch.cuda.device_count(),'vllm':vllm.__version__}))"],
                               capture_output=True, text=True)
        info = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else {"error": probe.stderr[-1500:]}
        step("import_check", ok=probe.returncode == 0 and info.get("gpus", 0) >= 2, **info)
        if probe.returncode != 0 or info.get("gpus", 0) < 2:
            raise RuntimeError("dynamo/vllm import or GPU check failed")
        rec["versions"] = info

        sys.path[:0] = [os.path.join(a.src, "src"), a.src]
        from bench.metrics import snapshot
        from bench.replay import build_items, load_trace, replay, summarize

        kv_dir = "/tmp/dyn_file_kv"
        base_env = dict(os.environ, DYN_DISCOVERY_BACKEND="file", DYN_FILE_KV=kv_dir, PYTHONHASHSEED="0",
                        HF_HUB_OFFLINE="1")

        def start_cluster(router_mode: str) -> None:
            shutil.rmtree(kv_dir, ignore_errors=True)
            for i in range(2):
                env = dict(base_env, CUDA_VISIBLE_DEVICES=str(i), DYN_SYSTEM_PORT=str(8091 + i))
                cmd = [vpy, "-m", "dynamo.vllm", "--model", model_dir, "--served-model-name", SERVED,
                       "--enable-prefix-caching", "--dtype", "half", "--max-model-len", "8192",
                       "--gpu-memory-utilization", "0.88"]
                procs.append(subprocess.Popen(cmd, env=env, stdout=open(os.path.join(out, f"worker{i}.log"), "a"),
                                              stderr=subprocess.STDOUT, start_new_session=True))
            cmd = [vpy, "-m", "dynamo.frontend", "--http-port", str(FRONT_PORT), "--router-mode", router_mode,
                   "--discovery-backend", "file"]
            procs.append(subprocess.Popen(cmd, env=base_env, stdout=open(os.path.join(out, f"frontend_{router_mode}.log"), "a"),
                                          stderr=subprocess.STDOUT, start_new_session=True))
            import urllib.request
            while True:
                box.check(f"cluster startup ({router_mode})")
                if any(p.poll() is not None for p in procs):
                    raise RuntimeError(f"a dynamo process exited during startup ({router_mode}); see worker*/frontend logs")
                with contextlib.suppress(Exception):
                    with urllib.request.urlopen(f"http://127.0.0.1:{FRONT_PORT}/v1/models", timeout=5) as r:
                        if SERVED in r.read().decode():
                            return
                time.sleep(5)

        def scrape() -> dict:
            import urllib.request
            texts = {}
            for name, port in (("worker0", 8091), ("worker1", 8092), ("frontend", FRONT_PORT)):
                with contextlib.suppress(Exception):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
                        texts[name] = r.read().decode("utf-8", "replace")
            return texts

        items = build_items(load_trace(os.path.join(run_dir, "bench_main.jsonl")), SERVED, 64, a.limit)
        plan = [("a_rr_raw", "round-robin", False), ("b_kv_raw", "kv", False), ("c_kv_l1", "kv", True)]
        for cid, mode, use_proxy in plan:
            box.check(f"condition {cid}")
            t0 = time.time()
            start_cluster(mode)
            step(f"{cid}:cluster_up", seconds=round(time.time() - t0))
            before = scrape()
            url = f"http://127.0.0.1:{FRONT_PORT}"
            proxy = None
            if use_proxy:
                env = dict(base_env, PREFIXANVIL_UPSTREAM_BASE_URL=url, PREFIXANVIL_PORT="8210", PREFIXANVIL_MODE="L1",
                           PREFIXANVIL_LOG=os.path.join(out, f"proxy_{cid}.jsonl"),
                           PYTHONPATH=os.pathsep.join([os.path.join(a.src, "src"), a.src]))
                proxy = subprocess.Popen([vpy, "-m", "prefixanvil", "proxy"], env=env, start_new_session=True,
                                         stdout=open(os.path.join(out, "proxy.out"), "a"), stderr=subprocess.STDOUT)
                time.sleep(3)
                url = "http://127.0.0.1:8210"
            t1 = time.time()
            results = replay(items, url, concurrency=1)
            wall = time.time() - t1
            after = scrape()
            if proxy:
                os.killpg(proxy.pid, signal.SIGTERM)
            for k, v in after.items():
                open(os.path.join(out, f"metrics_{cid}_{k}.txt"), "w").write(v)
            per = {}
            for k in after:
                sb, sa = snapshot(before.get(k, "")), snapshot(after[k])
                per[k] = {m: (None if sa["values"][m] is None or sb["values"][m] is None
                              else sa["values"][m] - sb["values"][m]) for m in ("prefix_cache_hits", "prefix_cache_queries")}
                per[k]["names"] = sa["names"]
            workers = [per.get(w, {}) for w in ("worker0", "worker1")]
            hits = sum(w.get("prefix_cache_hits") or 0 for w in workers)
            queries = sum(w.get("prefix_cache_queries") or 0 for w in workers)
            s = summarize(results, wall)
            rec["conditions"][cid] = {"router_mode": mode, "proxy_l1": use_proxy, "requests": len(items),
                                      "wall_s": wall, "per_endpoint": per,
                                      "worker_counters_found": all(w.get("prefix_cache_hits") is not None for w in workers),
                                      "hit_rate_workers_sum": hits / queries if queries else None,
                                      "usage_cached_tokens": s.cached_tokens, "usage_prompt_tokens": s.prompt_tokens,
                                      "errors": s.errors, "ttft_p50_s": s.ttft_p50_s, "ttft_p95_s": s.ttft_p95_s}
            step(f"{cid}:done", hit_rate=rec["conditions"][cid]["hit_rate_workers_sum"], errors=s.errors)
            stop_all()
        rec["status"] = "complete"
        return 0
    except TimeoutError as e:
        rec["status"] = "timebox_exhausted"
        rec["stopped_at"] = str(e)
        return 0
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:3000]
        return 0
    finally:
        stop_all()
        save()
        z = shutil.make_archive(os.path.join(os.path.abspath(a.workdir), f"prefixanvil_dynamo_{a.run_id}"), "zip", out)
        print(f"[dynamo] status={rec['status']} -> {z}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
