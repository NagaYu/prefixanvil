"""Render the analysis dict as JSON and Markdown."""

from __future__ import annotations

import json
from typing import Any


def write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _n(x: int | float) -> str:
    return f"{x:,.0f}" if isinstance(x, (int, float)) else str(x)


def to_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    c = r["conditions"]
    i, ii, iii = c["i_raw"], c["ii_l1"], c["iii_upper_bound"]
    tk = r["tokenizer"]
    sim = r["simulator"]
    tp = r["throughput_assumption"]

    a("# PrefixAnvil 解析レポート (Phase 1 / prefixanvil analyze)")
    a("")
    a(f"- トレース: `{r['trace']}`")
    a(f"- prefixanvil {r['prefixanvil_version']}")
    a("")
    a("> **注意:** 本レポートの数値はブロック・シミュレータによる *仕様に沿った近似* であり、"
      "vLLM の再現ではありません。実機との誤差は未検証です（Phase 3 で検証予定）。"
      "前提パラメータは末尾「前提」節を参照してください。")
    if tk.get("tokenizer_is_fixture"):
        a("")
        a("> **警告:** テスト用 fixture tokenizer で解析しています。実モデルのトークン数・"
          "ブロック境界とは一致しません。実ログには `--tokenizer <モデルの tokenizer ディレクトリ>` を指定してください。")
    if tk.get("template_uses_strftime_now"):
        a("")
        a("> **警告:** chat template が `strftime_now` を使用しています。レンダリング時刻依存の揮発情報 "
          "(F1 相当) がテンプレート内部に存在し、ログ上は見えません。")
    a("")

    a("## 1. サマリ")
    a("")
    a("| 条件 | リクエスト | プロンプト tokens | 再利用 tokens | 再計算 prefill tokens | ヒット率 (token 加重) | prefill GPU 秒 (概算) |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for name, d in (("(i) 実測相当: 正規化なし再生", i), ("(ii) L1 正規化後の再生", ii),
                    ("(iii) 参考上限 ※達成可能値ではない", iii)):
        a(f"| {name} | {_n(d['requests'])} | {_n(d['prompt_tokens'])} | {_n(d['cached_tokens'])} | "
          f"{_n(d['prefill_tokens'])} | {_pct(d['hit_rate_token_weighted'])} | {d['prefill_gpu_seconds_est']:.2f} |")
    s = r["savings_l1"]
    a("")
    a(f"- L1 による再計算 prefill 削減: **{_n(s['prefill_tokens_saved'])} tokens "
      f"({_pct(s['prefill_reduction_ratio'])})**, 概算 {s['gpu_seconds_saved_est']:.2f} GPU 秒")
    a("- (iii) は **上限であり達成可能値ではありません**。各リクエストが同一セッション内の最長の先行プロンプトを"
      "完全に延長し、セッション間で静的 prefix (system + tools) を共有できたと仮定し、"
      "キャッシュ容量無限・退避なしで計算しています。")
    a("- 各条件のヒット率はその条件自身のプロンプト tokens を分母にしています（L1 はバイト列を変えるため"
      "プロンプト長が条件 i と ii で僅かに異なります）。")
    a(f"- GPU 秒は prefill スループット **{tp['prefill_tokens_per_s']:,.0f} tokens/s を仮定** した線形概算です"
      "（実測値ではありません。attention の二次項やバッチングは考慮していません）。")
    a("")

    ing = r["ingest"]
    a("## 2. 取り込み")
    a("")
    a(f"- 総行数 {ing['total_lines']} / 空行 {ing['blank_lines']} / 受理 {ing['accepted']} / "
      f"不正 {ing['rejected_total']} / レンダリング失敗 {sum(ing['render_failures'].values())} / 再生 {ing['replayed']}")
    a(f"- 再生順: `{ing['replay_order']}`")
    if ing["rejected_by_reason"]:
        a("")
        a("| 不正理由 | 件数 | 行番号 (例) |")
        a("|---|---:|---|")
        for k, v in ing["rejected_by_reason"].items():
            a(f"| `{k}` | {v} | {', '.join(map(str, ing['rejected_sample_lines'].get(k, [])))} |")
    if ing["render_failures"]:
        a("")
        a("| レンダリング失敗 | 件数 |")
        a("|---|---:|")
        for k, v in ing["render_failures"].items():
            a(f"| `{k}` | {v} |")
    a("")

    p = r["pairs"]
    a("## 3. 原因別内訳 (正規化なし, トークン損失の大きい順)")
    a("")
    a(f"比較ペア {p['total']} 組 (scope: {p['by_scope']}), 断片化あり {p['fragmented_raw']} 組 → "
      f"L1 後 {p['fragmented_after_l1']} 組。")
    a("")
    a("損失 tokens = 参照リクエストと比べ、安定 prefix なら再利用できたはずのブロック量 − 実際の共通 prefix "
      "のブロック量（ペア単位の推定。キャッシュ退避の影響は含まない）。")
    a("")
    _cause_table(L, r["causes_raw"])
    a("")
    a("### L1 適用後に残る原因 (residual)")
    a("")
    _cause_table(L, r["causes_residual_after_l1"])
    a("")

    a("## 4. first divergence の分布 (正規化なし)")
    a("")
    fd = r["first_divergence"]
    a("| トークン位置 | ペア数 |")
    a("|---|---:|")
    for k, v in fd["by_token_position"].items():
        a(f"| {k} | {v} |")
    a("")
    a("| プロンプト内相対位置 | ペア数 |")
    a("|---|---:|")
    for k, v in fd["by_relative_position"].items():
        a(f"| {k} | {v} |")
    a("")
    a("シミュレータ上の最初のミス位置 (token 換算, 条件 i):")
    a("")
    a("| 位置 | リクエスト数 |")
    a("|---|---:|")
    for k, v in r["first_miss_block_raw"].items():
        a(f"| {k} | {v} |")
    a("")

    l1 = r["l1_transforms"]
    a("## 5. L1 正規化")
    a("")
    a(f"- 変換されたリクエスト: {l1['requests_touched']}")
    a("")
    a("| ルール | 適用回数 |")
    a("|---|---:|")
    for rule in l1["rules"]:
        a(f"| `{rule}` | {l1['applied_by_rule'].get(rule, 0)} |")
    if l1["skipped_fail_open"]:
        a("")
        a("| 素通り (fail-open) | 件数 |")
        a("|---|---:|")
        for k, v in l1["skipped_fail_open"].items():
            a(f"| `{k}` | {v} |")
    a("")

    if "classifier_evaluation" in r:
        ev = r["classifier_evaluation"]
        a("## 6. 分類器の評価 (AgentLoad 正解ラベル)")
        a("")
        a("| F | support | predicted | TP | 再現率 | 適合率 |")
        a("|---|---:|---:|---:|---:|---:|")
        for f, d in ev["per_fault"].items():
            a(f"| {f} | {d['support']} | {d['predicted']} | {d['tp']} | {_pct(d['recall'])} | {_pct(d['precision'])} |")
        a("")
        a("混同行列 (行=正解, 列=予測):")
        a("")
        cols = sorted({k for row in ev["confusion"].values() for k in row})
        a("| 正解 \\ 予測 | " + " | ".join(cols) + " |")
        a("|---|" + "---:|" * len(cols))
        for t, row in ev["confusion"].items():
            a(f"| {t} | " + " | ".join(str(row.get(k, 0)) for k in cols) + " |")
        a("")

    a("## 前提")
    a("")
    a(f"- tokenizer: `{tk['tokenizer']}`")
    a(f"- chat template: 出所 `{tk['template_source']}`, バージョン `{tk['template_version']}`")
    a(f"- ブロックサイズ {sim['block_size']} tokens, キャッシュ容量 {sim['num_blocks']:,} ブロック "
      f"({sim['capacity_tokens']:,} tokens), LRU 退避, model 名をハッシュに含める: {sim['hash_model_name']}")
    a("- 完全ブロックのみキャッシュ対象。親ハッシュ連鎖で最初のミスで再利用停止。"
      "最終トークンは常に再計算（vLLM V1 と同様）。")
    a("- リクエストは 1 件ずつ逐次処理（並行実行・バッチング・chunked prefill の干渉なし）。"
      "decode トークン（応答）はキャッシュに積まない。ログ順に再生。")
    a("- 原因分類はリクエスト対（同一セッションの直前 / セッション先頭は直前セッションの先頭）の "
      "first divergence に対するルールベース推定。推定できないものは `unknown`。")
    a("")
    return "\n".join(L)


def _cause_table(L: list[str], causes: list[dict]) -> None:
    if not causes:
        L.append("(断片化は検出されませんでした)")
        return
    L.append("| 原因 | 説明 | ペア数 | 損失 tokens | 損失シェア | 主な detail |")
    L.append("|---|---|---:|---:|---:|---|")
    for cdict in causes:
        det = ", ".join(f"{k} ({v})" for k, v in list(cdict["details"].items())[:3])
        L.append(f"| **{cdict['label']}** | {cdict['description']} | {cdict['pairs']} | "
                 f"{cdict['lost_tokens']:,} | {_pct(cdict['lost_share'])} | {det} |")


def write_markdown(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
