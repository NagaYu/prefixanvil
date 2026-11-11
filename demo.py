"""End-to-end demo on fixture data: prefixanvil gen -> prefixanvil analyze.

    .venv/bin/python demo.py [--out-dir out/demo]

Fully offline: uses the bundled fixture tokenizer (NOT a real model tokenizer),
so absolute token counts are illustrative only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from prefixanvil.cli import main as prefixanvil  # noqa: E402


def run(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    trace = os.path.join(out_dir, "trace.jsonl")
    labels = os.path.join(out_dir, "trace.labels.jsonl")
    report_md = os.path.join(out_dir, "report.md")
    cfg = os.path.join(ROOT, "fixtures", "configs", "demo.json")
    print(f"$ prefixanvil gen --config {os.path.relpath(cfg, ROOT)} --out {trace}")
    prefixanvil(["gen", "--config", cfg, "--out", trace])
    print(f"$ prefixanvil analyze --trace {trace} --out {report_md} --labels {labels}")
    prefixanvil(["analyze", "--trace", trace, "--out", report_md, "--labels", labels])
    with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    print()
    print("top causes (raw):")
    for c in report["causes_raw"][:8]:
        print(f"  {c['label']:8s} pairs={c['pairs']:3d} lost_tokens={c['lost_tokens']:6d} "
              f"({c['lost_share']:.0%})  {c['description']}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out", "demo"))
    run(ap.parse_args().out_dir)
