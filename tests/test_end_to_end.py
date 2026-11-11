import glob
import json
import os

import pytest

from prefixanvil.agentload import AgentLoadConfig, generate, write_trace
from prefixanvil.cli import main as prefixanvil
from prefixanvil.pipeline import AnalyzeOptions, analyze
from prefixanvil.report import to_markdown


def fixture_configs(root):
    return sorted(glob.glob(os.path.join(root, "fixtures", "configs", "*.json")))


def test_committed_fixture_traces_are_reproducible(root, tmp_path):
    cfgs = fixture_configs(root)
    assert len(cfgs) == 10
    for c in cfgs:
        name = os.path.splitext(os.path.basename(c))[0]
        with open(c) as f:
            cfg = AgentLoadConfig.from_dict(json.load(f))
        reqs, labels = generate(cfg)
        out = tmp_path / f"{name}.jsonl"
        write_trace(reqs, labels, str(out), cfg=cfg)
        committed = os.path.join(root, "fixtures", "traces", f"{name}.jsonl")
        assert out.read_bytes() == open(committed, "rb").read(), name


@pytest.mark.parametrize("fault", [f"F{i}" for i in range(1, 9)])
def test_single_fault_fixture_is_attributed(root, fault):
    t = os.path.join(root, "fixtures", "traces", f"only_{fault.lower()}.jsonl")
    rep = analyze(t, AnalyzeOptions(labels=t.replace(".jsonl", ".labels.jsonl")))
    causes = [c["label"] for c in rep["causes_raw"]]
    assert causes and causes[0] == fault           # the injected fault dominates
    assert rep["classifier_evaluation"]["per_fault"][fault]["recall"] >= 0.8


def test_clean_fixture_has_no_fragmentation(root):
    rep = analyze(os.path.join(root, "fixtures", "traces", "clean.jsonl"))
    assert rep["causes_raw"] == []
    c = rep["conditions"]
    assert rep["pairs"]["fragmented_after_l1"] == 0
    # L1 still rewrites (tool sort, key sort) which shifts token counts / block
    # alignment slightly, so (i) and (ii) agree only approximately
    hi, hii = c["i_raw"]["hit_rate_token_weighted"], c["ii_l1"]["hit_rate_token_weighted"]
    assert abs(hi - hii) < 0.005
    assert c["iii_upper_bound"]["hit_rate_token_weighted"] >= c["ii_l1"]["hit_rate_token_weighted"]


def test_l1_fixable_faults_improve_hit_rate(root):
    for f in ("f2", "f6"):
        rep = analyze(os.path.join(root, "fixtures", "traces", f"only_{f}.jsonl"))
        c = rep["conditions"]
        assert c["ii_l1"]["hit_rate_token_weighted"] > c["i_raw"]["hit_rate_token_weighted"], f
        assert rep["savings_l1"]["prefill_tokens_saved"] > 0
        assert rep["pairs"]["fragmented_after_l1"] < rep["pairs"]["fragmented_raw"]


def test_malformed_fixture_reports_rejections(root):
    rep = analyze(os.path.join(root, "fixtures", "traces", "malformed.jsonl"))
    ing = rep["ingest"]
    assert ing["accepted"] == 3 and ing["rejected_total"] == 5 and ing["blank_lines"] == 1
    assert set(ing["rejected_by_reason"]) == {"invalid_json", "empty_messages", "missing_model",
                                              "invalid_role", "invalid_timestamp"}
    md = to_markdown(rep)
    assert "invalid_json" in md


def test_bounds_ordering_and_report_fields(root):
    rep = analyze(os.path.join(root, "fixtures", "traces", "demo.jsonl"))
    c = rep["conditions"]
    assert c["iii_upper_bound"]["hit_rate_token_weighted"] >= c["ii_l1"]["hit_rate_token_weighted"]
    assert "UPPER BOUND" in c["iii_upper_bound"]["note"]
    assert rep["throughput_assumption"]["prefill_tokens_per_s"] == 10000.0
    assert rep["tokenizer"]["template_version"].startswith("sha256:")
    assert rep["simulator"]["block_size"] == 16
    md = to_markdown(rep)
    for needle in ("仕様に沿った近似", "fixture tokenizer", "参考上限", "tokens/s を仮定", "(iii)"):
        assert needle in md


def test_analysis_is_deterministic(root):
    t = os.path.join(root, "fixtures", "traces", "demo.jsonl")
    assert analyze(t) == analyze(t)


def test_cli_gen_and_analyze(root, tmp_path, capsys):
    cfg = os.path.join(root, "fixtures", "configs", "only_f3.json")
    trace = tmp_path / "t.jsonl"
    assert prefixanvil(["gen", "--config", cfg, "--out", str(trace), "--seed", "9"]) == 0
    assert (tmp_path / "t.labels.jsonl").exists()
    out = tmp_path / "r.md"
    assert prefixanvil(["analyze", "--trace", str(trace), "--out", str(out), "--block-size", "32",
                  "--cache-blocks", "64", "--labels", str(tmp_path / "t.labels.jsonl")]) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["simulator"]["block_size"] == 32 and rep["simulator"]["num_blocks"] == 64
    assert rep["conditions"]["i_raw"]["evictions"] > 0          # tiny cache forces LRU eviction
    assert "classifier_evaluation" in rep
    assert "hit rate (i) raw" in capsys.readouterr().out


def test_demo_runs(root, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo", os.path.join(root, "demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    rep = demo.run(str(tmp_path))
    assert (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()
    assert rep["conditions"]["i_raw"]["requests"] > 0
