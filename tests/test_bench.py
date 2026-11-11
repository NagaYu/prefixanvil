"""Phase 3 Step 0: measurement harness, verified offline against a fake vLLM server."""

import contextlib
import json
import os
import sys

import pytest

from bench import fidelity
from bench.equiv_sample import run as equiv_run
from bench.fake_vllm import FakeVLLM
from bench.free_gpu import run_all
from bench.metrics import cache_config, diff, parse_prometheus, snapshot
from bench.replay import build_items, load_trace, replay, summarize
from bench.runner import Condition, RunConfig, Runner, load_plan
from bench.vllm_ctl import metrics_text, reset_prefix_cache, verify_reset
from prefixanvil.agentload import AgentLoadConfig, generate, write_trace
from prefixanvil.equiv import run_suite
from prefixanvil.proxy import ConfigError, AnvilProxy, ProxyConfig
from prefixanvil.render import load_tokenizer, vllm_normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EQUIV = os.path.join(ROOT, "fixtures", "equiv")


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    cfg = AgentLoadConfig.from_dict({"seed": 5, "sessions": 3, "turns": 2, "system_tokens": 200, "tools": 4,
                                     "faults": {"F1": 0.15, "F2": 0.15, "F3": 0.15, "F6": 0.15}})
    reqs, labels = generate(cfg)
    p = d / "t.jsonl"
    write_trace(reqs, labels, str(p))
    return str(p)


@contextlib.contextmanager
def inproc_proxy(upstream, report=None):
    @contextlib.contextmanager
    def factory(cond):
        cfg = ProxyConfig(upstream=upstream, port=0, mode=cond.proxy_mode, l1_tool_order=cond.tool_order,
                          l2_enable=cond.proxy_mode == "L2", equiv_report=report or "missing.json")
        px = AnvilProxy(cfg).start()
        try:
            yield px.url
        finally:
            px.stop()
    yield factory


# --------------------------------------------------------------------------- metrics
def test_metric_resolution_current_legacy_missing():
    for style, name in (("v033", "vllm:prefix_cache_hits_total"), ("legacy", "vllm:gpu_prefix_cache_hits_total")):
        f = FakeVLLM(name_style=style)
        f.c.update(hits=40, queries=100)
        s = snapshot(f.metrics())
        assert s["names"]["prefix_cache_hits"] == name and s["values"]["prefix_cache_hits"] == 40.0
        assert "prefix_cache_hits" not in s["missing"]
    s = snapshot(FakeVLLM(name_style="missing").metrics())
    assert {"prefix_cache_hits", "prefix_cache_queries"} <= set(s["missing"])
    assert s["values"]["prefill_kv_computed_sum"] == 0.0
    cc = cache_config(s)
    assert cc["block_size"] == 16 and cc["num_gpu_blocks"] == 4096


def test_parse_and_diff():
    t1 = 'a_total{x="1"} 3\na_total{x="2"} 4\n# HELP x\nvllm:prefix_cache_hits_total{e="0"} 10\nvllm:prefix_cache_queries_total 40\n'
    t2 = t1.replace(" 10", " 30").replace(" 40", " 80")
    assert parse_prometheus(t1)[0].labels == (("x", "1"),)
    d = diff(snapshot(t1), snapshot(t2))
    assert d["prefix_cache_hits"] == 20 and d["prefix_cache_queries"] == 40 and d["hit_rate"] == 0.5


# --------------------------------------------------------------------------- replay
def test_build_items_bodies_and_predecessors(trace):
    lines = load_trace(trace)
    items = build_items(lines, "pf-model", max_tokens=7)
    b = items[0].body
    assert b["model"] == "pf-model" and b["temperature"] == 0 and b["max_tokens"] == 7 and b["stream"] is True
    assert not {"request_id", "session_id", "timestamp"} & set(b)
    for it in items:
        if it.predecessor is not None:
            assert lines[it.predecessor]["session_id"] == it.session_id and it.predecessor < it.index


def test_replay_sequential_and_concurrent(trace):
    items = build_items(load_trace(trace), "pf-model", max_tokens=4)
    with FakeVLLM() as f:
        r1 = replay(items, f.url, concurrency=1)
        assert [r.index for r in r1] == [i.index for i in items]
        assert all(r.status == 200 and r.ttft_s is not None and r.prompt_tokens for r in r1)
        s = summarize(r1, 1.0)
        assert s.errors == 0 and s.per_request_cached_available and s.cached_tokens > 0
        r8 = replay(items, f.url, concurrency=8)
        by = {r.index: r for r in r8}
        for it in items:                              # session order preserved under concurrency
            if it.predecessor is not None:
                p = by[it.predecessor]
                assert by[it.index].started_at >= p.started_at + p.total_s - 0.05


# --------------------------------------------------------------------------- reset
def test_reset_and_verification():
    with FakeVLLM() as f:
        v = verify_reset(f.url, "pf-model")
        assert v["verified"] and v["warm_hits"] > 0 and v["hits_after_reset"] == 0
    with FakeVLLM(dev_mode=False) as f:
        assert reset_prefix_cache(f.url)["status"] == "unsupported"


# --------------------------------------------------------------------------- runner
def test_runner_plan_resume_and_files(trace, tmp_path):
    rep = run_suite(EQUIV)
    rp = tmp_path / "equivalence-report.json"
    rp.write_text(json.dumps(rep))
    out = tmp_path / "res"
    with FakeVLLM() as f, inproc_proxy(f.url, str(rp)) as factory:
        cfg = RunConfig(f.url, "pf-model", str(out), max_tokens=4, versions={"vllm": "fake"})
        conds = [Condition("a_raw", trace, passes=2, warmup="full"),
                 Condition("b_l1", trace, target="proxy", proxy_mode="L1", passes=1, warmup=3),
                 Condition("c_l2", trace, target="proxy", proxy_mode="L2", passes=1, warmup="none")]
        res = Runner(cfg, factory, log=lambda m: None).run_plan(conds)
        assert res["conditions"] == {"a_raw": "complete", "b_l1": "complete", "c_l2": "complete"}
        a = json.loads((out / "a_raw.json").read_text())
        assert a["status"] == "complete" and len(a["passes"]) == 2 and "results" not in a["warmup"]
        p1, p2 = a["passes"]
        assert p1["reset"]["verify"]["verified"]
        # the cache is reset between passes -> both passes see identical accounting
        assert p1["counters"]["prefix_cache_hits"] == p2["counters"]["prefix_cache_hits"] > 0
        assert p1["counters"]["prefix_cache_hits"] == p1["summary"]["cached_tokens"]
        assert a["versions"] == {"vllm": "fake"}
        resets = f.resets
        Runner(cfg, factory, log=lambda m: None).run_plan(conds)          # resume: nothing re-run
        assert f.resets == resets
        b = json.loads((out / "b_l1.json").read_text())
        assert b["passes"][0]["counters"]["hit_rate"] >= p1["counters"]["hit_rate"]


def test_runner_restarts_server_when_reset_unsupported(trace, tmp_path):
    restarts = []
    with FakeVLLM(dev_mode=False) as f:
        cfg = RunConfig(f.url, "pf-model", str(tmp_path), max_tokens=2, limit=4)
        r = Runner(cfg, restart_server=lambda: (restarts.append(1), f.reset()), log=lambda m: None)
        r.run_condition(Condition("x", trace, passes=1, warmup="none"))
    assert restarts == [1]


def test_budget_downgrades_are_recorded(trace, tmp_path):
    with FakeVLLM() as f:
        cfg = RunConfig(f.url, "pf-model", str(tmp_path), max_tokens=2, budget_s=1.0)
        r = Runner(cfg, log=lambda m: None)
        r.s_per_request = 10.0                     # pretend requests are slow
        rec = r.run_condition(Condition("req", trace, passes=2, warmup="full"))
        assert rec["effective"]["passes"] == 1 and rec["effective"]["warmup"] == 30
        r.s_per_request = 10.0                     # the pass above re-measured the (fast) fake
        opt = r.run_condition(Condition("opt", trace, passes=1, warmup="none", optional=True))
        assert opt["status"] == "skipped_budget"
        assert any("2 -> 1" in d for d in r.decisions) and any("skipped" in d for d in r.decisions)


def test_missing_metrics_keep_raw_text(trace, tmp_path):
    with FakeVLLM(name_style="missing") as f:
        cfg = RunConfig(f.url, "pf-model", str(tmp_path), max_tokens=2, limit=3)
        r = Runner(cfg, log=lambda m: None)
        r.reset = lambda: {"reset": "skipped-in-test"}       # verification needs the hit counter
        rec = r.run_condition(Condition("m", trace, passes=1, warmup="none"))
    p = rec["passes"][0]
    assert "prefix_cache_hits" in p["missing_metrics"] and "raw_metrics_after" in p
    assert p["counters"]["hit_rate"] is None


def test_load_plan_committed():
    conds = load_plan(os.path.join(ROOT, "fixtures", "bench", "plan_t4.json"), "/tmp/x")
    ids = [c.id for c in conds]
    assert ids[:3] == ["main_a_raw", "main_b_l1", "main_c_l2"]
    assert all(c.trace.startswith("/tmp/x/") for c in conds)
    assert {c.tool_order for c in conds if c.id.startswith("xs_")} == {"session", "stateless"}


# --------------------------------------------------------------------------- fidelity
def test_fidelity_exact_against_fake(trace, tmp_path):
    with FakeVLLM(block_size=16, num_blocks=4096) as f:
        cfg = RunConfig(f.url, "pf-model", str(tmp_path), max_tokens=2)
        rec = Runner(cfg, log=lambda m: None).run_condition(Condition("raw", trace, passes=1, warmup="none"))
    m = fidelity.measured_pass(rec, 1)
    tok = load_tokenizer()
    row = fidelity.compare(m, fidelity.simulate(trace, tok, "raw", "vllm", 16, 4096), 16)
    assert row["hit_rate_diff_pts"] == 0 and row["prefill_diff_pct"] == 0
    assert row["cached_exact"] == row["requests"] == row["prompt_tokens_exact"]



def test_client_rendering_differs_when_tool_keys_are_shuffled(tmp_path):
    """vLLM re-serializes tools; a client that sends {"parameters", "name"} order is
    rendered differently by the client-bytes simulator than by vLLM."""
    tool = {"function": {"parameters": {"type": "object"}, "description": "d", "name": "f"}, "type": "function"}
    sysmsg = {"role": "system", "content": "You are a careful agent. " * 20}
    lines = [{"model": "m", "session_id": "s", "messages": [sysmsg, {"role": "user", "content": f"q{i}"}],
              "tools": [tool]} for i in range(3)]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    with FakeVLLM() as f:
        rec = Runner(RunConfig(f.url, "pf-model", str(tmp_path), max_tokens=2), log=lambda m: None) \
            .run_condition(Condition("k", str(p), passes=1, warmup="none"))
    m = fidelity.measured_pass(rec, 1)
    tok = load_tokenizer()
    v = fidelity.compare(m, fidelity.simulate(str(p), tok, "raw", "vllm", 16, 4096), 16)
    c = fidelity.compare(m, fidelity.simulate(str(p), tok, "raw", "client", 16, 4096), 16)
    assert v["prompt_tokens_exact"] == 3 and c["prompt_tokens_exact"] < 3


def test_token_change_reports_both_renderings(trace):
    tc = fidelity.token_change(trace, load_tokenizer())
    assert set(tc) == {"client", "vllm"}
    assert tc["vllm"]["keysort_change"] == pytest.approx(0, abs=0.02)   # vLLM re-serializes tools itself
    assert tc["client"]["raw_tokens"] > 0


def test_vllm_normalize_matches_source_behaviour():
    tools = [{"function": {"parameters": {"b": 1, "a": 2}, "name": "f"}, "type": "function"},
             {"type": "function", "function": {"name": "g", "description": "d", "strict": True}}]
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"y": 1,  "x": 2}'}},
        {"id": "2", "type": "function", "function": {"name": "f", "arguments": "not json"}},
        {"id": "3", "type": "function", "function": {"name": "f", "arguments": "[1]"}}]},
            {"role": "assistant", "content": "x", "tool_calls": []}]
    m, t = vllm_normalize(msgs, tools)
    assert list(t[0]) == ["type", "function"] and list(t[0]["function"]) == ["name", "description", "parameters"]
    assert t[0]["function"]["description"] is None and list(t[0]["function"]["parameters"]) == ["b", "a"]
    assert t[1]["function"]["strict"] is True
    args = [tc["function"]["arguments"] for tc in m[0]["tool_calls"]]
    assert args == [{"y": 1, "x": 2}, {}, {}] and list(args[0]) == ["y", "x"]
    assert "tool_calls" not in m[1]
    assert msgs[0]["tool_calls"][0]["function"]["arguments"].startswith("{")           # input untouched


# --------------------------------------------------------------------------- equivalence sampler
def test_equiv_sampler_against_fake():
    with FakeVLLM() as f:
        rep = equiv_run(EQUIV, f.url, "pf-model", max_tokens=8)
    s = rep["summary"]
    assert s["errors"] == 0
    assert s["control_raw_vs_raw"]["rate"] == 1.0 and s["warm_cache_vs_cold"]["rate"] == 1.0
    assert s["L1_vs_raw"]["rate"] == 1.0
    l2 = [r for r in rep["rows"] if r["case"] == "l2_volatile_lines"][0]["L2"]
    assert l2["match"] is False and l2["auto_tag"] in ("formatting_only", "needs_review") and "text_a" in l2
    stored = [r for r in rep["rows"] if r.get("control", {}).get("match")]
    assert all("text_a" not in r["control"] for r in stored)          # texts only for mismatches


# --------------------------------------------------------------------------- runner script + knobs
def test_run_all_pin_choice_and_platform(monkeypatch, tmp_path):
    assert run_all._choose_pins("13.0", None) == ["cu13", "cu129"]
    assert run_all._choose_pins("12.4", None) == ["cu129", "cu13"]
    assert run_all._choose_pins(None, "cu13") == ["cu13"]
    assert run_all.VLLM_PINS["cu13"]["spec"] == "vllm==0.33.0"
    assert run_all.EXPECTED_LICENSE[run_all.PRIMARY_MODEL] == "apache-2.0"
    assert run_all.FALLBACK_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert run_all.EXPECTED_LICENSE[run_all.FALLBACK_MODEL] == "apache-2.0"
    assert run_all.fallback_model(run_all.PRIMARY_MODEL, False, False) == run_all.FALLBACK_MODEL
    assert run_all.fallback_model(run_all.FALLBACK_MODEL, False, False) is None      # retry once only
    assert run_all.fallback_model(run_all.PRIMARY_MODEL, False, True) is None        # --no-fallback
    assert run_all.fallback_model(run_all.PRIMARY_MODEL, True, False) is None        # fake server
    # behaviour: the primary server fails once -> fallback downloaded and served, switch recorded
    import types
    calls = []

    class C:
        a = types.SimpleNamespace(model=run_all.PRIMARY_MODEL, fake_server=False, no_fallback=False)
        state = {"model": {"repo": run_all.PRIMARY_MODEL}}
        path = staticmethod(lambda *p: str(tmp_path.joinpath(*p)))
        save = log = staticmethod(lambda *x: None)

    def st_server(c, srv):
        calls.append(("server", c.state["model"]["repo"]))
        if c.state["model"]["repo"] == run_all.PRIMARY_MODEL:
            raise RuntimeError("vLLM did not become healthy (awq)")

    def st_model(c, model=None):
        calls.append(("model", model))
        c.state["model"] = {"repo": model}
    monkeypatch.setattr(run_all, "st_server", st_server)
    monkeypatch.setattr(run_all, "st_model", st_model)
    c = C()
    run_all.start_server_with_fallback(c, types.SimpleNamespace(stop=lambda: calls.append(("stop",))))
    assert calls == [("server", run_all.PRIMARY_MODEL), ("stop",), ("model", run_all.FALLBACK_MODEL),
                     ("server", run_all.FALLBACK_MODEL)]
    assert c.state["fallback"]["from"] == run_all.PRIMARY_MODEL and "awq" in c.state["fallback"]["reason"]


def test_agentload_client_tool_order():
    base = {"seed": 9, "sessions": 6, "turns": 1, "tools": 6, "faults": {}}
    fixed, _ = generate(AgentLoadConfig.from_dict(base))
    shuf, _ = generate(AgentLoadConfig.from_dict({**base, "client_tool_order": "per_session_shuffle"}))
    names = lambda r: tuple(t["function"]["name"] for t in r["tools"])
    first = {}
    for r in shuf:
        first.setdefault(r["session_id"], names(r))
        assert names(r) == first[r["session_id"]]                    # constant within a session
    assert len(set(first.values())) > 1                              # differs across sessions
    assert len({names(r) for r in fixed}) == 1
    # the main RNG stream is untouched: same request count and system prompts (which tool the
    # agent calls may differ, because tools are picked by position in the client's list)
    assert len(fixed) == len(shuf)
    assert [r["messages"][0] for r in fixed] == [r["messages"][0] for r in shuf]
    with pytest.raises(ValueError):
        AgentLoadConfig.from_dict({**base, "client_tool_order": "random"})


def test_proxy_tool_order_env():
    cfg = ProxyConfig.from_env({"PREFIXANVIL_UPSTREAM_BASE_URL": "http://127.0.0.1:1", "PREFIXANVIL_L1_TOOL_ORDER": "stateless"})
    assert cfg.l1_tool_order == "stateless"
    with pytest.raises(ConfigError):
        ProxyConfig.from_env({"PREFIXANVIL_UPSTREAM_BASE_URL": "http://127.0.0.1:1", "PREFIXANVIL_L1_TOOL_ORDER": "x"})


def test_st_install_parses_probe_output_while_install_output_goes_to_log(monkeypatch, tmp_path):
    """Regression for Kaggle smoke run t4-20261109-1422: every install step logs to install.log,
    so a probe run the same way had stdout=None and `.strip()` failed for BOTH pins although
    the probe printed valid JSON. The fake below behaves like subprocess.run: output redirected
    to a file is written there and NOT returned; output to PIPE is returned."""
    import subprocess
    import types

    kaggle_cu13 = ('{"torch": "2.14.0+cu132", "torch_cuda": "13.2", "cuda_available": true, '
                   '"capability": [7, 5], "vllm": "0.33.0", "transformers": "5.20.2"}')
    calls = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, env=None, timeout=None, **kw):
        is_probe = "-c" in cmd and "cuda_available" in cmd[-1]
        calls.append(("probe" if is_probe else cmd[2] if len(cmd) > 2 else cmd[0], stdout is subprocess.PIPE))
        out = (kaggle_cu13 + "\n") if is_probe else "Installed 1 package\n"
        if stdout is subprocess.PIPE:
            return subprocess.CompletedProcess(cmd, 0, out, "UserWarning: something at import\n"
                                               if stderr is subprocess.PIPE else None)
        stdout.write(out)                                   # redirected: lands in the log only
        return subprocess.CompletedProcess(cmd, 0, None, None)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    a = types.SimpleNamespace(workdir=str(tmp_path), src=str(tmp_path / "src"), run_id="r",
                              venv=str(tmp_path / "venv"), fake_server=False, vllm_pin=None)
    c = run_all.Ctx(a)
    c.state["env"] = {"driver_cuda": "13.0"}                # the Kaggle driver -> cu13 pin first
    run_all.st_install(c)

    inst = c.state["install"]
    assert inst["status"] == "ok" and inst["pin"] == "cu13" and inst["tried"] == [{"pin": "cu13", "ok": True}]
    assert inst["versions"] == json.loads(kaggle_cu13)
    assert ("probe", True) in calls                         # the probe's stdout was captured, not redirected
    log = (tmp_path / "results" / "r" / "install.log").read_text()
    assert kaggle_cu13 in log and "[captured stderr]" in log   # audit trail kept in install.log


def test_build_reexec_argv_drops_run_id_value_in_both_forms():
    """Regression for Kaggle run t4-20261110-0159: the old filter removed the --run-id flag
    token but left its value, so the re-exec'd child saw a bare positional and argparse exited 2."""
    f = run_all.build_reexec_argv

    # space form: NEITHER --run-id nor its value survives in the middle of the tail
    argv = f("/venv/py", "/s/run_all.py",
             ["--src", "/s", "--run-id", "t4-20261110-0159", "--profile", "smoke"], "t4-20261110-0159", 123.5)
    assert argv[:2] == ["/venv/py", "/s/run_all.py"]
    assert "t4-20261110-0159" not in argv[2:-4]          # orphaned value gone from the body
    assert argv.count("--run-id") == 1 and argv[-4:] == ["--run-id", "t4-20261110-0159", "--t0", "123.5"]
    assert argv[2:-4] == ["--src", "/s", "--profile", "smoke"]

    # equals form: --run-id=X removed as one token
    argv = f("/p", "/s", ["--run-id=abc", "--src", "/s"], "abc", 1.0)
    assert "--run-id=abc" not in argv and argv[2:-4] == ["--src", "/s"]

    # no --run-id present: body unchanged, canonical pair appended once
    argv = f("/p", "/s", ["--src", "/s", "--profile", "full"], "rid", 9.0)
    assert argv[2:] == ["--src", "/s", "--profile", "full", "--run-id", "rid", "--t0", "9.0"]

    # other flags' values are preserved verbatim; an existing --t0 is also dropped (not doubled)
    argv = f("/p", "/s", ["--max-model-len", "6144", "--t0", "5.0", "--budget-hours", "9.5"], "rid", 2.0)
    assert argv[2:] == ["--max-model-len", "6144", "--budget-hours", "9.5", "--run-id", "rid", "--t0", "2.0"]
    assert argv.count("--t0") == 1

    # the exact notebook shape: every token in the child body is a flag or a flag's value
    # (no stray positional like the Kaggle "t4-20261110-0159" that argparse rejected)
    tail = ["--src", "/s", "--workdir", "/w", "--run-id", "t4-x", "--profile", "smoke", "--budget-hours", "9.5"]
    body = f(sys.executable, "/s/run_all.py", tail, "t4-x", 100.0)[2:]
    i = 0
    while i < len(body):
        assert body[i].startswith("--"), f"stray positional {body[i]!r} would make argparse exit 2"
        i += 1 if "=" in body[i] else 2


def test_serve_command_enables_auto_tool_choice():
    """Regression for Kaggle run t4-20261110-2339: the bench traces carry `tools`, so vLLM's
    OpenAI server resolved tool_choice to "auto" and returned HTTP 400 for all 12 smoke requests
    ("auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set).
    The fake vLLM does not enforce this, so the flags are pinned here directly."""
    import types
    c = types.SimpleNamespace(
        state={"model": {"local_dir": "/models/m"}},
        venv="/venv",
        a=types.SimpleNamespace(max_model_len=8192, gpu_mem_util=0.9))
    cmd = run_all.Server(c).cmd()
    assert "--enable-auto-tool-choice" in cmd
    i = cmd.index("--tool-call-parser")
    assert cmd[i + 1] == "hermes"                       # Qwen2.5 format (primary 7B-AWQ + 1.5B fallback)
    assert cmd[:2] == ["/venv/bin/vllm", "serve"] and "--enable-prefix-caching" in cmd
