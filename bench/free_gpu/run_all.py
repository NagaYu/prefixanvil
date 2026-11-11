#!/usr/bin/env python3
"""PrefixAnvil Phase 3 Step 1 — one-shot runner for free-tier GPU notebooks (Kaggle / Colab).

Called once from a notebook cell. Stages (each recorded in ``stages.json``;
completed stages are skipped on re-run, so a cut session can be resumed):

  env       platform + GPU + driver/CUDA facts (nvidia-smi saved verbatim)
  install   isolated venv (uv), pinned vLLM chosen from the driver's CUDA version,
            PrefixAnvil from the embedded source; import + CUDA check
  model     Hugging Face snapshot (revision sha recorded), license checked
  server    vllm serve with prefix caching, prompt-token details, dev-mode reset;
            startup log, `--help` and the resolved cache config recorded
  traces    AgentLoad traces regenerated from the committed configs (sha256 recorded)
  smoke     12-request direct + proxy check before spending hours
  equiv     `prefixanvil equiv` on this machine (L2 needs a report for *this* build)
  plan      the measurement plan (bench.runner), resumable per condition
  sample    real-model equivalence sampler
  fidelity  simulator with the real tokenizer / block size / KV capacity
  package   summary.md + results zip

Nothing here creates accounts, enters credentials or uses paid services. The only
network use is pip/uv installs and the model download from Hugging Face.
Stdlib only until the venv exists; after `install` the script re-executes itself
inside the venv.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Fallback when the AWQ model cannot be served on this GPU. Qwen2.5-3B-Instruct was the original choice
# but its card says license "other" / "qwen-research"; replaced by the Apache-2.0 1.5B model (2026-11-07).
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_LICENSE = {PRIMARY_MODEL: "apache-2.0", FALLBACK_MODEL: "apache-2.0"}
VLLM_PINS = {
    # verified 2026-11: PyPI wheels of 0.33.0 are CUDA 13 builds (torch 2.14, *[cu13] deps);
    # 0.32.0 is the newest release with a CUDA 12.9 build on wheels.vllm.ai.
    "cu13": {"spec": "vllm==0.33.0", "args": ["--torch-backend=auto"]},
    "cu129": {"spec": "vllm==0.32.0+cu129",
              "args": ["--extra-index-url", "https://wheels.vllm.ai/0.32.0/cu129",
                       "--torch-backend=cu129", "--index-strategy", "unsafe-best-match"]},
}
SERVED = "pf-model"
PORT = 8000
STAGES = ["env", "install", "model", "server", "traces", "smoke", "equiv", "plan", "sample", "fidelity", "package"]


# --------------------------------------------------------------------------- helpers
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.work = os.path.abspath(a.workdir)
        self.src = os.path.abspath(a.src)
        self.run_dir = os.path.join(self.work, "results", a.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.state_path = os.path.join(self.run_dir, "stages.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.venv = a.venv
        self.vpy = sys.executable if a.fake_server else os.path.join(self.venv, "bin", "python")
        self.port = PORT
        self.base = f"http://127.0.0.1:{PORT}"
        self.logf = open(os.path.join(self.run_dir, "run_all.log"), "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def save(self) -> None:
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_path)

    def done(self, stage: str) -> bool:
        ok = self.state.get(stage, {}).get("status") == "ok"
        # machine-local stages are only "done" if their artefacts exist on *this* machine
        # (a resumed Kaggle version runs on a fresh VM: /tmp is empty)
        if ok and stage == "install" and not self.a.fake_server:
            ok = os.path.exists(self.vpy)
        if ok and stage == "model":
            ok = os.path.isdir(self.state["model"].get("local_dir", ""))
        return ok

    def record(self, stage: str, status: str, **info) -> None:
        self.state[stage] = {"status": status, "at": now(), **info}
        self.save()

    def path(self, *p: str) -> str:
        return os.path.join(self.run_dir, *p)


def sh(cmd: list[str], log_path: str | None = None, env: dict | None = None, timeout: float | None = None,
       check: bool = True) -> subprocess.CompletedProcess:
    """Run a command. With ``log_path`` the output goes to that file and the returned
    ``stdout`` is None — use :func:`sh_capture` when the output must also be parsed."""
    out = open(log_path, "a") if log_path else subprocess.PIPE
    try:
        p = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout)
    finally:
        if log_path:
            out.close()
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)[:300]}"
                           + (f" — see {log_path}" if log_path else f"\n{p.stdout[-2000:]}"))
    return p


def sh_capture(cmd: list[str], log_path: str, env: dict | None = None,
               timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command with stdout and stderr captured (PIPE) so the caller can parse
    stdout, then append both to ``log_path`` to keep the audit trail."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=timeout)
    with open(log_path, "a") as f:
        f.write(f"[captured] exit={p.returncode}: {' '.join(cmd)[:120]}\n")
        f.write(p.stdout or "")
        if p.stderr:
            f.write("[captured stderr]\n" + p.stderr)
        if (p.stdout or p.stderr) and not (p.stdout + p.stderr).endswith("\n"):
            f.write("\n")
    return p


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None or os.path.isdir("/content"):
        return "colab"
    return "other"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_ok(url: str, timeout: float = 5.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- stages
def st_env(c: Ctx) -> None:
    smi = shutil.which("nvidia-smi")
    info = {"platform": detect_platform(), "python": sys.version, "machine": platform.platform(),
            "cpu_count": os.cpu_count(), "disk_free_gb": shutil.disk_usage(c.work).free / 1e9}
    if not smi:
        c.record("env", "failed", **info, error="nvidia-smi not found: GPU accelerator is not enabled")
        raise SystemExit("No GPU. Kaggle: Settings > Accelerator > GPU T4 x2. Colab: Runtime > Change runtime type > T4.")
    full = sh([smi], check=False).stdout
    open(c.path("nvidia-smi.txt"), "w").write(full)
    q = sh([smi, "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
           check=False).stdout.strip()
    gpus = [dict(zip(["index", "name", "memory_total", "driver", "compute_cap"], [x.strip() for x in line.split(",")]))
            for line in q.splitlines() if line.strip()]
    m = re.search(r"CUDA Version:\s*([0-9.]+)", full)
    cuda = m.group(1) if m else None
    info.update(gpus=gpus, driver_cuda=cuda, internet=http_ok("https://pypi.org/simple/pip/", 10))
    if not info["internet"]:
        c.record("env", "failed", **info, error="no internet")
        raise SystemExit("Internet is off. Kaggle: Settings > Internet ON (needs a phone-verified account).")
    c.record("env", "ok", **info)
    for g in gpus:
        c.log(f"GPU {g['index']}: {g['name']} {g['memory_total']} driver {g['driver']} cc {g['compute_cap']}")
    c.log(f"driver CUDA {cuda}; platform {info['platform']}")


def _choose_pins(cuda: str | None, forced: str | None) -> list[str]:
    if forced:
        return [forced]
    try:
        major = int(float(cuda or "0"))
    except ValueError:
        major = 0
    return ["cu13", "cu129"] if major >= 13 else ["cu129", "cu13"]


def st_install(c: Ctx) -> None:
    log = c.path("install.log")
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"], log)
    uv = [sys.executable, "-m", "uv"]
    if not os.path.exists(c.vpy):
        sh(uv + ["venv", c.venv, "--python", "3.12", "--seed"], log)
    tried = []
    for key in _choose_pins(c.state["env"].get("driver_cuda"), c.a.vllm_pin):
        pin = VLLM_PINS[key]
        c.log(f"installing {pin['spec']} ({key}) ... this takes several minutes")
        try:
            sh(uv + ["pip", "install", "--python", c.vpy, pin["spec"], *pin["args"],
                     "huggingface_hub", "pytest"], log, timeout=3600)
            sh(uv + ["pip", "install", "--python", c.vpy, "-e", c.src], log, timeout=900)
            probe = sh_capture([c.vpy, "-c", "import json,torch,vllm,transformers;print(json.dumps({'torch':torch.__version__,"
                        "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
                        "'capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,"
                        "'vllm':vllm.__version__,'transformers':transformers.__version__}))"], log)
            info = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else {}
            if not info.get("cuda_available"):
                raise RuntimeError(f"torch cannot use the GPU with {pin['spec']}: {info or 'import failed'}")
            tried.append({"pin": key, "ok": True})
            c.record("install", "ok", pin=key, spec=pin["spec"], versions=info, tried=tried)
            c.log(f"installed: {info}")
            return
        except Exception as e:
            tried.append({"pin": key, "ok": False, "error": str(e)[:500]})
            c.log(f"install with {key} failed: {e}")
    c.record("install", "failed", tried=tried)
    raise SystemExit("vLLM install failed for all pins — see install.log in the results zip")


def st_model(c: Ctx, model: str | None = None) -> None:
    from huggingface_hub import HfApi, snapshot_download
    model = model or c.a.model
    api = HfApi()
    mi = api.model_info(model)
    cd = getattr(mi, "card_data", None)
    cdd = cd.to_dict() if hasattr(cd, "to_dict") else dict(cd or {})
    lic = cdd.get("license")
    if lic == "other":                 # e.g. a card with license "other" + license_name "qwen-research"
        lic = cdd.get("license_name") or "other"
    expected = EXPECTED_LICENSE.get(model)
    info = {"repo": model, "revision": mi.sha, "license": lic, "expected_license": expected}
    if expected and lic != expected:
        c.record("model", "stopped", **info, error="license differs from the expected value")
        raise SystemExit(f"STOP: {model} license is {lic!r}, expected {expected!r}. Not continuing (report it).")
    local = snapshot_download(model, revision=mi.sha, local_dir=os.path.join(c.a.model_dir, model.replace("/", "__")))
    c.record("model", "ok", **info, local_dir=local)
    c.log(f"model {model}@{mi.sha[:12]} license {lic} -> {local}")


def fallback_model(current: str, fake: bool, disabled: bool) -> str | None:
    """Which model to retry with after the server failed to start with `current` (None = give up)."""
    if fake or disabled or current == FALLBACK_MODEL:
        return None
    return FALLBACK_MODEL


def start_server_with_fallback(c: Ctx, srv) -> None:
    """Start vLLM with the recorded model; if it cannot be served (e.g. AWQ kernels on this GPU),
    record why, download the Apache-2.0 fallback and retry once. Never silent: the switch is in
    stages.json["fallback"], in versions() of every condition, and in BENCH.md."""
    try:
        st_server(c, srv)
        return
    except Exception as e:
        current = c.state.get("model", {}).get("repo", c.a.model)
        nxt = fallback_model(current, c.a.fake_server, c.a.no_fallback)
        if nxt is None:
            raise
        srv.stop()
        if os.path.exists(c.path("vllm_server.log")):
            os.replace(c.path("vllm_server.log"), c.path("vllm_server_primary_failed.log"))
        c.state["fallback"] = {"from": current, "to": nxt, "at": now(),
                               "reason": f"{type(e).__name__}: {e}"[-2000:]}
        c.save()
        c.log(f"FALLBACK: {current} could not be served ({type(e).__name__}); switching to {nxt}")
    os.environ.pop("HF_HUB_OFFLINE", None)
    st_model(c, nxt)
    st_server(c, srv)


class FakeServer:
    """--fake-server: the offline fake vLLM (bench.fake_vllm) instead of `vllm serve`."""

    def __init__(self, c: Ctx):
        self.c, self.fake = c, None

    def cmd(self) -> list[str]:
        return ["bench.fake_vllm.FakeVLLM"]

    def start(self) -> None:
        from bench.fake_vllm import FakeVLLM
        self.fake = FakeVLLM(served_model=SERVED).__enter__()
        self.c.base = self.fake.url

    def stop(self) -> None:
        if self.fake:
            self.fake.__exit__()
            self.fake = None

    def restart(self) -> None:
        self.stop()
        self.start()


class Server:
    def __init__(self, c: Ctx):
        self.c, self.proc = c, None

    def cmd(self) -> list[str]:
        m = self.c.state["model"]["local_dir"]
        return [os.path.join(self.c.venv, "bin", "vllm"), "serve", m, "--served-model-name", SERVED,
                "--port", str(PORT), "--host", "127.0.0.1", "--enable-prefix-caching",
                "--enable-prompt-tokens-details", "--dtype", "half", "--max-model-len", str(self.c.a.max_model_len),
                "--gpu-memory-utilization", str(self.c.a.gpu_mem_util), "--seed", "0",
                # the bench traces carry tools; with tools present and no explicit tool_choice, vLLM's
                # OpenAI server resolves tool_choice to "auto" and 400s unless these are set. "hermes" is
                # the Qwen2.5 tool-call format (covers the 7B-AWQ primary and the 1.5B fallback). A real
                # deployment serving agent traffic must run this way too.
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

    def start(self) -> None:
        env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="0",
                   HF_HUB_OFFLINE="1")
        logp = self.c.path("vllm_server.log")
        self.c.log("starting vLLM: " + " ".join(self.cmd()))
        self.proc = subprocess.Popen(self.cmd(), stdout=open(logp, "a"), stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        from bench.vllm_ctl import wait_healthy
        if not wait_healthy(self.c.base, timeout_s=1800, alive=lambda: self.proc.poll() is None):
            tail = open(logp).read()[-3000:]
            raise RuntimeError(f"vLLM did not become healthy (exit={self.proc.poll()}). Log tail:\n{tail}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def restart(self) -> None:
        self.stop()
        self.start()


def st_server(c: Ctx, srv: Server) -> None:
    helptxt = "" if c.a.fake_server else sh([os.path.join(c.venv, "bin", "vllm"), "serve", "--help=all"],
                                            check=False).stdout
    open(c.path("vllm_serve_help.txt"), "w").write(helptxt)
    srv.start()
    from bench.metrics import cache_config, snapshot
    from bench.vllm_ctl import metrics_text
    cc = cache_config(snapshot(metrics_text(c.base)))
    log = open(c.path("vllm_server.log")).read() if os.path.exists(c.path("vllm_server.log")) else ""
    backend = re.findall(r"Using (\S+) attention backend", log) or re.findall(r"Using (\S+) backend", log)
    quant = re.findall(r"(?i)(awq[_a-z]*)\b.*(?:kernel|method|quantization)", log)[:3]
    blk_help = [ln.strip() for ln in helptxt.splitlines() if "block-size" in ln or "block_size" in ln][:6]
    c.record("server", "ok", cmd=srv.cmd(), cache_config=cc, attention_backend=backend[-1] if backend else None,
             quantization_log=quant, block_size_help=blk_help)
    c.log(f"server up: block_size={cc['block_size']} num_gpu_blocks={cc['num_gpu_blocks']} backend={backend[-1:]}")


def st_traces(c: Ctx) -> None:
    import hashlib
    from prefixanvil.agentload import AgentLoadConfig, generate, write_trace
    out = {}
    for name in ("bench_main", "bench_xsession"):
        cfg = AgentLoadConfig.from_dict(json.load(open(os.path.join(c.src, "fixtures", "bench", f"{name}.json"))))
        reqs, labels = generate(cfg)
        p = c.path(f"{name}.jsonl")
        write_trace(reqs, labels, p, cfg=cfg)
        out[name] = {"requests": len(reqs), "labels": len(labels),
                     "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    c.record("traces", "ok", traces=out)


def versions(c: Ctx) -> dict:
    s = c.state
    return {"platform": s["env"]["platform"], "gpus": s["env"]["gpus"], "driver_cuda": s["env"]["driver_cuda"],
            "install": s["install"].get("versions"), "vllm_pin": s["install"].get("spec"),
            "model": {k: s["model"][k] for k in ("repo", "revision", "license")},
            "model_fallback": s.get("fallback"),
            "server_cmd": s["server"]["cmd"], "cache_config": s["server"]["cache_config"],
            "attention_backend": s["server"]["attention_backend"]}


@contextlib.contextmanager
def proxy_process(c: Ctx, cond):
    port = free_port()
    env = dict(os.environ, PREFIXANVIL_UPSTREAM_BASE_URL=c.base, PREFIXANVIL_PORT=str(port),
               PYTHONPATH=os.pathsep.join([os.path.join(c.src, "src"), c.src]),
               PREFIXANVIL_MODE=cond.proxy_mode, PREFIXANVIL_L1_TOOL_ORDER=cond.tool_order,
               PREFIXANVIL_LOG=c.path(f"proxy_{cond.id}.jsonl"), HF_HUB_OFFLINE="1")
    if cond.proxy_mode == "L2":
        env.update(PREFIXANVIL_L2_ENABLE="1", PREFIXANVIL_EQUIV_REPORT=c.path("equivalence-report.json"))
    p = subprocess.Popen([c.vpy, "-m", "prefixanvil", "proxy"], env=env, cwd=c.run_dir,
                         stdout=open(c.path(f"proxy_{cond.id}.out"), "a"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if p.poll() is not None:
                raise RuntimeError(f"prefixanvil proxy exited ({p.returncode}); see proxy_{cond.id}.out")
            if http_ok(f"http://127.0.0.1:{port}/health", 2):
                break
            time.sleep(0.5)
        yield f"http://127.0.0.1:{port}"
    finally:
        p.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(20)


def _runner(c: Ctx, srv: Server, out_dir: str, limit=None, budget=None):
    from bench.runner import RunConfig, Runner
    cfg = RunConfig(base_url=c.base, served_model=SERVED, out_dir=out_dir,
                    max_tokens=c.a.max_tokens, budget_s=budget, limit=limit, versions=versions(c))
    return Runner(cfg, proxy_factory=lambda cond: proxy_process(c, cond), restart_server=srv.restart, log=c.log)


def st_smoke(c: Ctx, srv: Server) -> None:
    from bench.runner import Condition
    r = _runner(c, srv, c.path("smoke"), limit=12)
    conds = [Condition("smoke_direct", c.path("bench_main.jsonl"), passes=1, warmup="none"),
             Condition("smoke_l1", c.path("bench_main.jsonl"), target="proxy", proxy_mode="L1", passes=1, warmup="none")]
    res = r.run_plan(conds)
    bad = {k: v for k, v in res["conditions"].items() if v != "complete"}
    if bad:
        c.record("smoke", "failed", result=res)
        raise SystemExit(f"smoke test failed: {bad}")
    for cid in ("smoke_direct", "smoke_l1"):
        p = json.load(open(c.path("smoke", f"{cid}.json")))["passes"][0]
        if p["summary"]["errors"]:
            c.record("smoke", "failed", result=res)
            raise SystemExit(f"smoke {cid}: {p['summary']['errors']} request errors")
    c.record("smoke", "ok", result=res)


def st_equiv(c: Ctx) -> None:
    out = c.path("equivalence-report.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(c.src, "src"), c.src]))
    p = sh([c.vpy, "-m", "prefixanvil", "equiv", "--suite", os.path.join(c.src, "fixtures", "equiv"), "--out", out],
           c.path("equiv.log"), env=env, check=False)
    c.record("equiv", "ok" if p.returncode == 0 else "failed", returncode=p.returncode)
    if p.returncode != 0:
        c.log("prefixanvil equiv FAILED: L2 condition will refuse to start (recorded)")


def st_plan(c: Ctx, srv: Server) -> None:
    from bench.runner import load_plan
    conds = load_plan(os.path.join(c.src, "fixtures", "bench", "plan_t4.json"), c.run_dir)
    if c.a.profile == "smoke":
        for x in conds:
            x.passes, x.warmup = 1, "none"
    elapsed = time.time() - c.a.t0
    budget = max(600.0, c.a.budget_hours * 3600 - elapsed - 1800)   # keep 30 min for sample/fidelity/package
    r = _runner(c, srv, c.path("conditions"), limit=40 if c.a.profile == "smoke" else None, budget=budget)
    try:
        res = r.run_plan(conds)
    finally:
        c.state.setdefault("plan", {})["decisions"] = r.decisions
        c.save()
    c.record("plan", "ok", result=res, decisions=r.decisions, budget_s=budget)


def st_sample(c: Ctx) -> None:
    from bench.equiv_sample import run
    rep = run(os.path.join(c.src, "fixtures", "equiv"), c.base, SERVED, c.a.max_tokens)
    json.dump(rep, open(c.path("equiv_sample.json"), "w"), indent=1, ensure_ascii=False)
    c.record("sample", "ok", summary=rep["summary"])


def st_fidelity(c: Ctx) -> None:
    from bench.fidelity import compare, measured_pass, simulate, token_change
    from prefixanvil.render import load_tokenizer
    tok = load_tokenizer(c.state["model"]["local_dir"])
    cc = c.state["server"]["cache_config"]
    bs, nb = cc["block_size"] or 16, cc["num_gpu_blocks"] or 8192
    rows, missing = [], []
    limit = 40 if c.a.profile == "smoke" else None      # must match what st_plan replayed
    pairs = [("main_a_raw", "bench_main.jsonl", "raw"), ("main_b_l1", "bench_main.jsonl", "L1"),
             ("xs_raw", "bench_xsession.jsonl", "raw"), ("xs_session", "bench_xsession.jsonl", "L1"),
             ("xs_stateless", "bench_xsession.jsonl", "L1-stateless")]
    for cid, trace, policy in pairs:
        p = c.path("conditions", f"{cid}.json")
        if not os.path.exists(p) or json.load(open(p)).get("status") != "complete":
            missing.append(cid)
            continue
        m = measured_pass(json.load(open(p)), 1)
        for rendering in ("client", "vllm"):
            rows.append({"condition": cid, **compare(m, simulate(c.path(trace), tok, policy, rendering, bs, nb,
                                                                  limit), bs)})
    tc = token_change(c.path("bench_main.jsonl"), tok, limit)
    json.dump({"block_size": bs, "num_gpu_blocks": nb, "limit": limit, "rows": rows, "missing_conditions": missing,
               "token_change": tc}, open(c.path("fidelity.json"), "w"), indent=1)
    c.record("fidelity", "ok", missing=missing)


def st_package(c: Ctx) -> None:
    c.record("package", "ok")
    lines = [f"# PrefixAnvil Phase 3 run {c.a.run_id}", "", f"- finished: {now()}", f"- profile: {c.a.profile}"]
    for k in STAGES:
        lines.append(f"- {k}: {c.state.get(k, {}).get('status', 'not run')}")
    dec = c.state.get("plan", {}).get("decisions") or []
    if dec:
        lines += ["", "## budget decisions", ""] + [f"- {d}" for d in dec]
    open(c.path("summary.md"), "w").write("\n".join(lines) + "\n")
    z = shutil.make_archive(os.path.join(c.work, f"prefixanvil_results_{c.a.run_id}"), "zip", c.run_dir)
    c.log(f"RESULTS ZIP: {z}")


# --------------------------------------------------------------------------- main
def build_reexec_argv(python: str, script: str, argv_tail: list[str], run_id: str, t0: float) -> list[str]:
    """argv to re-exec this script inside the venv. Drops any existing --run-id / --t0
    (both "--flag X" and "--flag=X" forms, VALUE included) from argv_tail, then appends
    one canonical --run-id/--t0 pair. Every other argument is kept verbatim and in order.

    The earlier filter dropped only the --run-id flag token and left its value orphaned,
    which argparse rejected as a stray positional in the re-exec'd child (Kaggle run
    t4-20261110-0159).
    """
    drop = ("--run-id", "--t0")
    out: list[str] = []
    skip_value = False
    for tok in argv_tail:
        if skip_value:
            skip_value = False
            continue
        if tok.split("=", 1)[0] in drop:
            skip_value = "=" not in tok          # space form: also drop the next token (the value)
            continue
        out.append(tok)
    return [python, script, *out, "--run-id", run_id, "--t0", str(t0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/kaggle/working/pf" if os.path.isdir("/kaggle/working") else "/content/pf"
                    if os.path.isdir("/content") else os.path.abspath("pf_work"))
    ap.add_argument("--src", required=True, help="PrefixAnvil source tree (extracted by the notebook)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--venv", default="/tmp/pfvenv")
    ap.add_argument("--model-dir", default="/tmp/pf_models")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--no-fallback", action="store_true", help="do not fall back to the 1.5B model")
    ap.add_argument("--vllm-pin", choices=list(VLLM_PINS), default=None)
    ap.add_argument("--profile", choices=["full", "smoke"], default="full")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--budget-hours", type=float, default=9.5)
    ap.add_argument("--fake-server", action="store_true",
                    help="offline dry run: fake vLLM + fixture tokenizer, skips env/install/model (no GPU)")
    ap.add_argument("--t0", type=float, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.run_id = a.run_id or dt.datetime.now(dt.timezone.utc).strftime("t4-%Y%m%d-%H%M")
    a.t0 = a.t0 or time.time()
    c = Ctx(a)
    c.log(f"run {a.run_id} profile={a.profile} python={sys.executable}")

    in_venv = os.path.realpath(sys.prefix) == os.path.realpath(c.venv)
    if a.fake_server:
        sys.path[:0] = [os.path.join(c.src, "src"), c.src]
        from prefixanvil.render import FIXTURE_TOKENIZER_DIR
        c.record("env", "ok", platform="fake", gpus=[], driver_cuda=None, python=sys.version)
        c.record("install", "ok", spec="(fake)", versions={"vllm": "fake"})
        c.record("model", "ok", repo="(fixture tokenizer)", revision="-", license="-", local_dir=FIXTURE_TOKENIZER_DIR)
    elif not in_venv:
        try:
            if c.state.get("env"):                      # always re-probe: a resume may be another VM
                c.state.setdefault("env_history", []).append(c.state["env"])
            st_env(c)
            if not c.done("install"):
                st_install(c)
        except SystemExit:
            st_package(c)          # diagnostics zip even when the environment is unusable
            raise
        argv = build_reexec_argv(c.vpy, os.path.abspath(__file__), sys.argv[1:], a.run_id, a.t0)
        c.log("re-executing inside the venv")
        os.execv(c.vpy, argv)

    sys.path[:0] = [os.path.join(c.src, "src"), c.src]
    if not c.done("model"):
        os.environ.pop("HF_HUB_OFFLINE", None)
        st_model(c, c.state.get("fallback", {}).get("to"))   # a resumed run keeps an earlier fallback
    srv = FakeServer(c) if a.fake_server else Server(c)
    try:
        if c.state.get("server"):
            c.state.setdefault("server_history", []).append(c.state["server"])
        start_server_with_fallback(c, srv)              # always: records this machine's cache config
        for name, fn in (("traces", lambda: st_traces(c)), ("smoke", lambda: st_smoke(c, srv)),
                         ("equiv", lambda: st_equiv(c)), ("plan", lambda: st_plan(c, srv)),
                         ("sample", lambda: st_sample(c))):
            if not c.done(name):
                c.log(f"=== stage {name}")
                try:
                    fn()
                except SystemExit:
                    raise
                except Exception as e:
                    c.record(name, "failed", error=f"{type(e).__name__}: {e}"[:2000])
                    c.log(f"stage {name} failed: {e}")
                    if name in ("traces", "smoke"):
                        raise
    finally:
        srv.stop()
        if not c.done("fidelity") and c.done("traces"):
            try:
                st_fidelity(c)
            except Exception as e:
                c.record("fidelity", "failed", error=f"{type(e).__name__}: {e}"[:2000])
        st_package(c)


if __name__ == "__main__":
    main()
