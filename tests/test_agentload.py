import json

import pytest

from prefixanvil.agentload import FAULTS, AgentLoadConfig, generate, load_labels, write_trace


def cfg(**kw):
    base = {"seed": 3, "sessions": 6, "turns": 2, "faults": {f: 0.1 for f in FAULTS}}
    base.update(kw)
    return AgentLoadConfig.from_dict(base)


def test_same_seed_same_bytes(tmp_path):
    outs = []
    for i in range(2):
        r, l = generate(cfg())
        p = tmp_path / f"t{i}.jsonl"
        write_trace(r, l, str(p))
        outs.append((p.read_bytes(), (tmp_path / f"t{i}.labels.jsonl").read_bytes()))
    assert outs[0] == outs[1]
    r2, _ = generate(cfg(seed=4))
    assert r2 != generate(cfg())[0]


def test_labels_point_at_real_requests():
    reqs, labels = generate(cfg(sessions=10))
    ids = {r["request_id"]: i for i, r in enumerate(reqs)}
    assert labels
    for lab in labels:
        assert reqs[lab["request_index"]]["request_id"] == lab["request_id"]
        assert lab["fault"] in FAULTS
        assert lab["reference_request_id"] in ids
        assert ids[lab["reference_request_id"]] < lab["request_index"]
        assert "element" in lab["location"]
        if lab["fault"] == "F4":
            assert lab["scope"] == "cross_session"
        if lab["fault"] == "F8":
            assert lab["scope"] == "within_session"


def test_zero_faults_zero_labels_and_timestamps_sorted():
    reqs, labels = generate(cfg(faults={}))
    assert labels == []
    ts = [r["timestamp"] for r in reqs]
    assert ts == sorted(ts)


def test_shape_parameters():
    reqs, _ = generate(cfg(sessions=3, turns=1, tools=0, tool_burst_min=0, tool_burst_max=0,
                           retry_rate=0.0, faults={}))
    assert len(reqs) == 3 and all("tools" not in r for r in reqs)
    reqs, _ = generate(cfg(sessions=2, turns=2, tool_burst_min=2, tool_burst_max=2,
                           retry_rate=0.0, faults={}))
    assert len(reqs) == 2 * 2 * 3


def test_retry_duplicates_content():
    reqs, _ = generate(cfg(retry_rate=1.0, sessions=1, turns=1, faults={}))
    a, b = reqs[0], reqs[1]
    assert a["messages"] == b["messages"] and a["request_id"] != b["request_id"]


def test_config_validation():
    with pytest.raises(ValueError):
        AgentLoadConfig.from_dict({"faults": {"F9": 0.1}})
    with pytest.raises(ValueError):
        AgentLoadConfig.from_dict({"faults": {f: 0.2 for f in FAULTS}})
    with pytest.raises(ValueError):
        AgentLoadConfig.from_dict({"bogus": 1})


def test_label_file_roundtrip(tmp_path):
    r, l = generate(cfg())
    p = tmp_path / "x.jsonl"
    lp = write_trace(r, l, str(p), cfg=cfg())
    assert load_labels(lp) == json.loads(json.dumps(l))
