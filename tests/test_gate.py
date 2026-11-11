"""prefixanvil gate: passes on the demo baseline, fails on a regressed trace, never rewrites implicitly."""

import json
import os
import shutil

import pytest

from prefixanvil.cli import main as prefixanvil
from prefixanvil.gate import compare, format_table

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE = os.path.join(ROOT, "fixtures", "gate", "demo-baseline.json")


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    d = tmp_path_factory.mktemp("gate")
    demo, reg = d / "demo.jsonl", d / "regressed.jsonl"
    assert prefixanvil(["gen", "--config", os.path.join(ROOT, "fixtures", "configs", "demo.json"), "--out", str(demo)]) == 0
    assert prefixanvil(["gen", "--config", os.path.join(ROOT, "fixtures", "gate", "demo_regressed_f1.json"),
                  "--out", str(reg)]) == 0
    return demo, reg


def gate(*args):
    return prefixanvil(["gate", *map(str, args)])


def test_passes_on_committed_demo_baseline(traces, capsys):
    before = open(BASELINE, "rb").read()
    assert gate("--trace", traces[0], "--baseline", BASELINE) == 0
    out = capsys.readouterr().out
    assert "| i_raw.hit_rate |" in out and "prefixanvil gate: PASS" in out
    assert open(BASELINE, "rb").read() == before                       # never modified implicitly


def test_fails_on_regressed_trace(traces, capsys):
    before = open(BASELINE, "rb").read()
    assert gate("--trace", traces[1], "--baseline", BASELINE) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "prefixanvil gate: FAIL" in out
    assert open(BASELINE, "rb").read() == before


def test_write_baseline_is_explicit(traces, tmp_path, capsys):
    b = tmp_path / "b.json"
    assert gate("--trace", traces[0], "--baseline", b) == 2              # missing baseline: error, not created
    assert not b.exists()
    assert gate("--trace", traces[0], "--baseline", b, "--write-baseline") == 0
    assert json.loads(b.read_text())["schema"] == "prefixanvil.gate-baseline/v1"
    shutil.copy(BASELINE, tmp_path / "copy.json")
    assert gate("--trace", traces[1], "--baseline", tmp_path / "copy.json", "--write-baseline") == 0
    assert gate("--trace", traces[1], "--baseline", tmp_path / "copy.json") == 0   # updated baseline now matches


def test_incomparable_settings_and_missing_trace(traces, tmp_path, capsys):
    assert gate("--trace", traces[0], "--baseline", BASELINE, "--block-size", "32") == 2
    assert "not comparable" in capsys.readouterr().out
    assert gate("--trace", tmp_path / "nope.jsonl", "--baseline", BASELINE) == 2


def test_thresholds_and_table():
    base = {"trace": {"sha256": "a"}, "metrics": {c: {"hit_rate": 0.70, "prefill_tokens": 1000}
                                                  for c in ("i_raw", "ii_l1")}}
    cur = {"trace": {"sha256": "b"}, "metrics": {"i_raw": {"hit_rate": 0.695, "prefill_tokens": 1015},
                                                 "ii_l1": {"hit_rate": 0.70, "prefill_tokens": 1000}}}
    r = compare(base, cur, 0.01, 0.02)
    assert r.passed and r.notes                                          # within limits; trace differs note
    assert not compare(base, cur, 0.004, 0.02).passed                    # 0.5 pt drop > 0.4 pt
    assert not compare(base, cur, 0.01, 0.01).passed                     # +1.5% prefill > 1%
    t = format_table(r.rows)
    assert t.count("\n") == 5 and "+1.50%" in t
