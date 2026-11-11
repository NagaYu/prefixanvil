"""Command line: ``prefixanvil gen | analyze | proxy | equiv | gate`` (argparse, stdlib only)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .agentload import AgentLoadConfig, generate, write_trace


def _cmd_gen(args: argparse.Namespace) -> int:
    with open(args.config, encoding="utf-8") as f:
        cfg = AgentLoadConfig.from_dict(json.load(f))
    if args.seed is not None:
        cfg.seed = args.seed
    reqs, labels = generate(cfg)
    lp = write_trace(reqs, labels, args.out, args.labels_out, cfg)
    print(f"wrote {len(reqs)} requests -> {args.out}")
    print(f"wrote {len(labels)} fault labels -> {lp}")
    return 0


def _analyze_opts(args: argparse.Namespace):
    from .classify import ClassifyConfig
    from .pipeline import AnalyzeOptions
    return AnalyzeOptions(
        tokenizer=args.tokenizer, chat_template=args.chat_template,
        allow_download=args.allow_download, block_size=args.block_size,
        num_blocks=args.cache_blocks, hash_model_name=args.hash_model_name,
        prefill_tokens_per_s=args.prefill_tps, order=args.order,
        classify=ClassifyConfig(rag_min_chars=args.rag_min_chars))


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import analyze
    from .report import write_json, write_markdown

    opts = _analyze_opts(args)
    opts.labels = args.labels
    report = analyze(args.trace, opts)
    json_out = args.json_out or os.path.splitext(args.out)[0] + ".json"
    write_markdown(report, args.out)
    write_json(report, json_out)
    c = report["conditions"]
    print(f"hit rate (i) raw: {c['i_raw']['hit_rate_token_weighted']:.1%}  "
          f"(ii) L1: {c['ii_l1']['hit_rate_token_weighted']:.1%}  "
          f"(iii) upper bound: {c['iii_upper_bound']['hit_rate_token_weighted']:.1%}")
    print(f"prefill reduction by L1: {report['savings_l1']['prefill_reduction_ratio']:.1%}")
    print(f"wrote {args.out} and {json_out}")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import main_from_env
    return main_from_env()


def _cmd_equiv(args: argparse.Namespace) -> int:
    from .equiv import run_suite, to_markdown
    rep = run_suite(args.suite)
    md_out = args.md_out or os.path.splitext(args.out)[0] + ".md"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(to_markdown(rep))
    for m, st in rep["summary"].items():
        rate = "n/a" if st["exact_match_rate"] is None else f"{st['exact_match_rate']:.1%}"
        print(f"{m}: {st['exact_match']}/{st['requests']} exact match ({rate}) over {st['cases']} cases")
    for a in rep["allowed_diffs"]:
        print(f"allowed diff: {a['case']} ({a['mode']})")
    for u in rep["unexpected_mismatches"]:
        print(f"UNEXPECTED MISMATCH: {u['case']} ({u['mode']}) first diff {u['first_diff']}")
    print(f"equivalence: {'PASSED' if rep['passed'] else 'FAILED'} -> {args.out}, {md_out}")
    return 0 if rep["passed"] else 1


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gate import run_gate
    return run_gate(args.trace, args.baseline, _analyze_opts(args), args.max_hitrate_drop,
                    args.max_prefill_growth, args.write_baseline)


def _add_analyzer_flags(a: argparse.ArgumentParser) -> None:
    a.add_argument("--tokenizer", default=None,
                   help="local tokenizer dir (default: bundled fixture tokenizer — NOT a real model)")
    a.add_argument("--chat-template", default=None, help="explicit Jinja chat template file")
    a.add_argument("--allow-download", action="store_true",
                   help="allow fetching --tokenizer from the HF hub (off by default)")
    a.add_argument("--block-size", type=int, default=16)
    a.add_argument("--cache-blocks", type=int, default=8192, help="KV cache capacity in blocks")
    a.add_argument("--hash-model-name", action="store_true", help="include model name in block hashes")
    a.add_argument("--prefill-tps", type=float, default=10000.0,
                   help="ASSUMED prefill throughput (tokens/s) for GPU-second estimates")
    a.add_argument("--order", choices=("file", "timestamp"), default="file")
    a.add_argument("--rag-min-chars", type=int, default=512, help="F7: minimum length of a 'long block'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prefixanvil", description="PrefixAnvil (Phase 2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a synthetic AgentLoad trace with fault labels")
    g.add_argument("--config", required=True, help="AgentLoad config (JSON)")
    g.add_argument("--out", required=True, help="output trace (JSONL)")
    g.add_argument("--labels-out", default=None, help="ground-truth labels (default: <out>.labels.jsonl)")
    g.add_argument("--seed", type=int, default=None, help="override config seed")
    g.set_defaults(func=_cmd_gen)

    a = sub.add_parser("analyze", help="replay a request log and report prefix-cache fragmentation")
    a.add_argument("--trace", required=True, help="request log (OpenAI-compatible JSONL)")
    a.add_argument("--out", required=True, help="Markdown report path (JSON written alongside)")
    a.add_argument("--json-out", default=None)
    a.add_argument("--labels", default=None, help="AgentLoad labels file -> classifier evaluation")
    _add_analyzer_flags(a)
    a.set_defaults(func=_cmd_analyze)

    x = sub.add_parser("proxy", help="run the OpenAI-compatible proxy (configured via PREFIXANVIL_* env vars)")
    x.set_defaults(func=_cmd_proxy)

    e = sub.add_parser("equiv", help="output-equivalence gate for L1/L2 against the deterministic mock upstream")
    e.add_argument("--suite", required=True, help="directory of prefixanvil.equiv-case/v1 JSON files")
    e.add_argument("--out", default="equivalence-report.json")
    e.add_argument("--md-out", default=None)
    e.set_defaults(func=_cmd_equiv)

    gt = sub.add_parser("gate", help="CI regression gate against a versioned baseline")
    gt.add_argument("--trace", required=True)
    gt.add_argument("--baseline", required=True)
    gt.add_argument("--max-hitrate-drop", type=float, default=0.01, help="absolute (0.01 = 1 point)")
    gt.add_argument("--max-prefill-growth", type=float, default=0.02, help="relative (0.02 = 2%%)")
    gt.add_argument("--write-baseline", action="store_true", help="create/update the baseline explicitly")
    _add_analyzer_flags(gt)
    gt.set_defaults(func=_cmd_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
