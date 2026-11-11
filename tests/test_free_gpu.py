"""The free-tier runner as the notebook will execute it: built notebook -> unpack cell ->
run_all.py --fake-server from the *unpacked* copy (proves the embedded source is complete)."""

import json
import os
import shutil
import subprocess
import sys

from bench.free_gpu import build_notebook


def test_notebook_unpacks_and_dry_runs(tmp_path):
    nb = build_notebook.build()
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(srcs) == 5 and "SRC_B64" in srcs[1] and "run_all.py" in srcs[2] and "run_dynamo.py" in srcs[3]
    g = {"WORK": str(tmp_path), "os": os, "shutil": shutil}
    exec(srcs[1], g)                                     # the notebook's own unpack cell
    src = g["SRC"]
    for rel in ("src/prefixanvil/core.py", "bench/free_gpu/run_all.py", "fixtures/bench/plan_t4.json",
                "fixtures/equiv/ws_code_block.json", "fixtures/tokenizer/tokenizer.json", "pyproject.toml"):
        assert os.path.exists(os.path.join(src, rel)), rel
    assert not os.path.exists(os.path.join(src, "bench", "results"))
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH="")
    p = subprocess.run([sys.executable, os.path.join(src, "bench/free_gpu/run_all.py"), "--fake-server",
                        "--src", src, "--workdir", str(tmp_path / "w"), "--run-id", "t", "--profile", "smoke"],
                       env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    run = tmp_path / "w" / "results" / "t"
    stages = json.loads((run / "stages.json").read_text())
    assert all(stages[k]["status"] == "ok" for k in ("traces", "smoke", "equiv", "plan", "sample", "fidelity", "package"))
    fid = json.loads((run / "fidelity.json").read_text())
    vllm_rows = [r for r in fid["rows"] if r["rendering"] == "vllm"]
    assert vllm_rows and all(r["hit_rate_diff_pts"] == 0 for r in vllm_rows)
    assert (tmp_path / "w" / "prefixanvil_results_t.zip").exists()
    # resume: a second invocation skips completed stages and re-packages
    p2 = subprocess.run([sys.executable, os.path.join(src, "bench/free_gpu/run_all.py"), "--fake-server",
                         "--src", src, "--workdir", str(tmp_path / "w"), "--run-id", "t", "--profile", "smoke"],
                        env=env, capture_output=True, text=True, timeout=600)
    assert p2.returncode == 0 and "=== stage plan" not in p2.stdout


def test_dynamo_runner_refuses_without_step1(tmp_path):
    from bench.free_gpu import run_dynamo
    os.makedirs(tmp_path / "results" / "r")
    sys_argv = sys.argv
    sys.argv = ["run_dynamo", "--workdir", str(tmp_path), "--src", ".", "--run-id", "r"]
    try:
        rc = run_dynamo.main()
    finally:
        sys.argv = sys_argv
    rec = json.loads((tmp_path / "results" / "r" / "dynamo" / "dynamo.json").read_text())
    assert rc == 2 and rec["status"] == "not_run" and "Step 1" in rec["steps"][0]["reason"]


def test_bench_md_from_results_refuses_fake(tmp_path):
    import pytest
    from bench.make_bench_md import build
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = subprocess.run([sys.executable, os.path.join(root, "bench/free_gpu/run_all.py"), "--fake-server", "--src", root,
                        "--workdir", str(tmp_path), "--run-id", "m", "--profile", "smoke"],
                       env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:]
    run = str(tmp_path / "results" / "m")
    with pytest.raises(SystemExit):
        build(run)                                       # fake results never become a BENCH.md
    md = build(run, allow_fake=True)
    for h in ("## Main result", "## Simulator fidelity", "## Real-model output equivalence", "T4 reference values",
              "## NVIDIA Dynamo", "not measured (no Dynamo run"):
        assert h in md, h
    assert "| (a) raw, direct to vLLM | 1 |" in md
