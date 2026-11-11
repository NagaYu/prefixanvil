"""`prefixanvil equiv`: output-equivalence gate for L1/L2 canonicalization.

Each fixture case (``prefixanvil.equiv-case/v1``) is a short session of
OpenAI chat requests. For every mode the case declares, each request is sent
to the upstream twice — raw and transformed through :func:`core.transform`
(with the caller-owned session context threaded through the session) — and
the two outputs are compared exactly. ``expect`` per mode is ``match`` (any
difference is a failure) or ``allowed-diff`` (a difference is permitted and
listed with the case's reason). The gate passes iff there is no unexpected
mismatch and no error.

By default the upstream is the in-process deterministic mock
(:mod:`prefixanvil.mock_upstream`); see its docstring for what it can and
cannot model.
"""

from __future__ import annotations

import glob
import hashlib
import http.client
import json
import os
from typing import Any, Callable
from urllib.parse import urlsplit

from . import __version__
from .core import TRANSFORM_REQUEST_SCHEMA, transform
from .canonical import SessionContext, canonicalize
from .mock_upstream import MODEL_ID, MockUpstream

CASE_SCHEMA = "prefixanvil.equiv-case/v1"
REPORT_SCHEMA = "prefixanvil.equivalence-report/v1"
EXPECT_VALUES = ("match", "allowed-diff")
_FINGERPRINT_MODULES = ("canonical.py", "relocate.py", "core.py", "classify.py", "ingest.py")


def build_fingerprint() -> str:
    """Identifies the transform code of the current build (L1/L2 + detectors)."""
    h = hashlib.sha256(__version__.encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _FINGERPRINT_MODULES:
        with open(os.path.join(here, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return "sha256:" + h.hexdigest()[:24]


def load_suite(suite_dir: str) -> tuple[list[dict], str]:
    paths = sorted(glob.glob(os.path.join(suite_dir, "*.json")))
    if not paths:
        raise ValueError(f"no *.json cases in {suite_dir}")
    cases, h = [], hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            raw = f.read()
        h.update(os.path.basename(p).encode() + b"\0" + raw)
        c = json.loads(raw)
        if c.get("schema") != CASE_SCHEMA:
            raise ValueError(f"{p}: schema must be {CASE_SCHEMA}")
        if not c.get("id") or not isinstance(c.get("requests"), list) or not c["requests"]:
            raise ValueError(f"{p}: needs id and non-empty requests")
        exp = c.get("expect") or {}
        if not exp or any(m not in ("L1", "L2") or v not in EXPECT_VALUES for m, v in exp.items()):
            raise ValueError(f"{p}: expect must map L1/L2 to {EXPECT_VALUES}")
        if "allowed-diff" in exp.values() and not c.get("allowed_diff_reason"):
            raise ValueError(f"{p}: allowed-diff needs allowed_diff_reason")
        c["_file"] = os.path.basename(p)
        cases.append(c)
    return cases, "sha256:" + h.hexdigest()[:24]


def _post(url: str, body: dict) -> str:
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    try:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn.request("POST", "/v1/chat/completions", data, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"upstream status {resp.status}")
        return json.loads(payload)["choices"][0]["message"]["content"]
    finally:
        conn.close()


def _first_diff(a: str, b: str) -> dict:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return {"char": i, "word": len(a[:i].split(" ")) - 1 if i else 0}


def run_suite(suite_dir: str, upstream_url: str | None = None,
              canonicalizer: Callable = canonicalize) -> dict:
    """Run the suite. ``canonicalizer`` is a test seam (inject a bad one to see the gate fail)."""
    cases, suite_hash = load_suite(suite_dir)
    mock = None
    if upstream_url is None:
        mock = MockUpstream().start()
        upstream_url = mock.url
    results, unexpected, allowed, errors = [], [], [], []
    stats = {m: {"cases": 0, "requests": 0, "exact_match": 0} for m in ("L1", "L2")}
    try:
        for c in cases:
            for mode, expect in sorted(c["expect"].items()):
                st = stats[mode]
                st["cases"] += 1
                ctx = SessionContext().to_json() if c.get("session", True) else None
                per_req = []
                for k, body in enumerate(c["requests"]):
                    entry: dict[str, Any] = {"index": k}
                    try:
                        res = transform({"schema": TRANSFORM_REQUEST_SCHEMA, "mode": mode,
                                         "request": body, "session_context": ctx},
                                        _canonicalize=canonicalizer)
                        ctx = res["session_context"]
                        out_raw = _post(upstream_url, body)
                        out_can = _post(upstream_url, res["request"])
                        entry.update(changed=res["changed"],
                                     rules=sorted({a["rule"] for a in res["applied"]}),
                                     skipped=sorted({f"{s['rule']}:{s['reason']}" for s in res["skipped"]}),
                                     match=out_raw == out_can)
                        if not entry["match"]:
                            entry["first_diff"] = _first_diff(out_raw, out_can)
                    except Exception as e:
                        entry.update(error=f"{type(e).__name__}: {e}", match=False)
                        errors.append({"case": c["id"], "mode": mode, "index": k, "error": entry["error"]})
                    st["requests"] += 1
                    st["exact_match"] += bool(entry.get("match"))
                    per_req.append(entry)
                matched = all(e.get("match") for e in per_req)
                outcome = "match" if matched else "diff"
                ok = matched or expect == "allowed-diff"
                rec = {"case": c["id"], "file": c["_file"], "mode": mode, "expect": expect,
                       "outcome": outcome, "ok": ok, "description": c.get("description", ""),
                       "requests": per_req}
                if expect == "allowed-diff":
                    rec["allowed_diff_reason"] = c["allowed_diff_reason"]
                    if not matched:
                        allowed.append({"case": c["id"], "mode": mode, "reason": c["allowed_diff_reason"]})
                if not ok:
                    unexpected.append({"case": c["id"], "mode": mode,
                                       "first_diff": next((e.get("first_diff") for e in per_req
                                                           if not e.get("match")), None)})
                results.append(rec)
    finally:
        if mock:
            mock.stop()
    for st in stats.values():
        st["exact_match_rate"] = st["exact_match"] / st["requests"] if st["requests"] else None
    return {
        "schema": REPORT_SCHEMA,
        "prefixanvil_version": __version__,
        "build_fingerprint": build_fingerprint(),
        "suite": {"path": suite_dir, "sha256": suite_hash, "cases": len(cases)},
        "upstream": {"kind": "mock" if mock else "external",
                     "equivalence_model": MODEL_ID if mock else None},
        "summary": stats,
        "allowed_diffs": allowed,
        "unexpected_mismatches": unexpected,
        "errors": errors,
        "passed": not unexpected and not errors,
        "results": results,
    }


def check_report(path: str, require_mode: str = "L2") -> tuple[bool, str]:
    """Is there a passing equivalence report for the *current* build?"""
    if not os.path.exists(path):
        return False, f"no equivalence report at {path} (run: prefixanvil equiv --suite fixtures/equiv/)"
    try:
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"unreadable equivalence report {path}: {type(e).__name__}"
    if rep.get("schema") != REPORT_SCHEMA:
        return False, f"equivalence report schema {rep.get('schema')!r} != {REPORT_SCHEMA}"
    if not rep.get("passed"):
        return False, "equivalence report did not pass"
    if rep.get("build_fingerprint") != build_fingerprint():
        return False, ("equivalence report is for a different build "
                       f"({rep.get('build_fingerprint')} != {build_fingerprint()}); re-run prefixanvil equiv")
    if not (rep.get("summary", {}).get(require_mode) or {}).get("requests"):
        return False, f"equivalence report contains no {require_mode} cases"
    return True, "ok"


def to_markdown(rep: dict) -> str:
    L = ["# PrefixAnvil equivalence report", "",
         f"- result: **{'PASSED' if rep['passed'] else 'FAILED'}**",
         f"- build: `{rep['build_fingerprint']}` (prefixanvil {rep['prefixanvil_version']})",
         f"- suite: `{rep['suite']['path']}` ({rep['suite']['cases']} cases, `{rep['suite']['sha256']}`)",
         f"- upstream: {rep['upstream']['kind']}"
         + (f" (`{rep['upstream']['equivalence_model']}` — deterministic mock, NOT a real model; "
            "real-model equivalence is a Phase 3 measurement)" if rep["upstream"]["kind"] == "mock" else ""),
         "", "| mode | cases | requests | exact match | rate |", "|---|---:|---:|---:|---:|"]
    for m, s in rep["summary"].items():
        rate = "n/a" if s["exact_match_rate"] is None else f"{s['exact_match_rate']:.1%}"
        L.append(f"| {m} | {s['cases']} | {s['requests']} | {s['exact_match']} | {rate} |")
    L += ["", "| case | mode | expect | outcome | ok |", "|---|---|---|---|---|"]
    for r in rep["results"]:
        L.append(f"| `{r['case']}` | {r['mode']} | {r['expect']} | {r['outcome']} | {'✓' if r['ok'] else '✗'} |")
    if rep["allowed_diffs"]:
        L += ["", "## Allowed diffs", ""]
        L += [f"- `{a['case']}` ({a['mode']}): {a['reason']}" for a in rep["allowed_diffs"]]
    if rep["unexpected_mismatches"]:
        L += ["", "## Unexpected mismatches", ""]
        L += [f"- `{u['case']}` ({u['mode']}): first diff {u['first_diff']}" for u in rep["unexpected_mismatches"]]
    if rep["errors"]:
        L += ["", "## Errors", ""] + [f"- `{e['case']}` ({e['mode']}#{e['index']}): {e['error']}" for e in rep["errors"]]
    return "\n".join(L) + "\n"
