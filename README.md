# PrefixAnvil

**PrefixAnvil detects and normalizes prefix / KV-cache fragmentation in agent LLM traffic.**
It sits in front of a self-hosted inference server (vLLM / TensorRT-LLM / NVIDIA Dynamo). Agent
frameworks emit requests whose prompts drift in ways that don't change meaning — reordered tool
definitions, volatile lines (timestamps, counters) spliced into an otherwise stable system prompt,
whitespace and JSON-serialization differences. Each drift breaks the longest shared prefix, so the
server's automatic prefix cache (APC) can't reuse earlier work and re-runs prefill it already paid
for. PrefixAnvil measures that loss, classifies its causes (F1–F8), and applies deterministic,
meaning-preserving normalization (L1), plus one volatile-line relocation (L2), to restore cache hits.

## Measured headline

Measured on **Kaggle free-tier 2 × Tesla T4**, model **Qwen2.5-7B-Instruct-AWQ**, vLLM 0.33.0,
AgentLoad synthetic traces (`bench_main`, concurrency 1). All numbers are machine-generated into
[`BENCH.md`](BENCH.md) from the run under `bench/results/t4-20261111-0408/`.

| metric (`bench_main`, token-weighted) | raw | L1 normalize | L2 relocate |
|---|---:|---:|---:|
| prefix-cache hit rate | 74.71% | 81.79% | 83.61% |
| recomputed prefill tokens | 241,640 | 172,268 (**−28.7%**) | 155,068 (**−35.8%**) |

Hit rate and prefill-token counts are determined by the cache algorithm and do **not** depend on the
GPU generation. **TTFT and GPU-seconds in `BENCH.md` are T4 reference values**, not representative of
H100-class GPUs; an H100 measurement is future work. The simulator is a spec-based approximation of
vLLM V1 APC, not a reproduction.

## Run it (offline, no GPU, no network)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # full suite, no GPU / no network
.venv/bin/python demo.py               # fixture gen → analyze, writes out/demo/
```

CLI without install: `PYTHONPATH=src .venv/bin/python -m prefixanvil {gen,analyze,proxy,equiv,gate} ...`.

## License

This repository is made **publicly visible solely so it can be read and evaluated**. It is **All
Rights Reserved** — no license is granted beyond viewing for evaluation. See [`LICENSE`](LICENSE).
Licensing or transfer inquiries: https://nagayu-portfolio.nagayu.workers.dev/en

---

The detailed sections below (in Japanese) document Phases 1–3 in full.

# PrefixAnvil — Phase 2

セルフホスト LLM 推論（vLLM / TensorRT-LLM / NVIDIA Dynamo）の手前で、エージェント・トラフィックの
KV / prefix キャッシュ断片化を **検出・正規化** する層です。

## ステータス: Phase 2

| コマンド | 内容 | Phase |
|---|---|---|
| `prefixanvil gen` / `prefixanvil analyze` | 合成トレース生成とオフライン解析（下記） | 1 |
| `prefixanvil proxy` | OpenAI 互換プロキシ（L0 観測 / L1 正規化 / L2 揮発行の移設） | 2 |
| `prefixanvil equiv` | 正規化が出力を変えないことを確認する等価性ゲート（決定的モック upstream） | 2 |
| `prefixanvil gate` | CI 回帰ゲート（ベースラインとのヒット率・prefill 比較） | 2 |
| `bench/` | 実 vLLM 計測ハーネスと無料枠 GPU ランナー（結果待ち） | 3 |

Phase 2 の詳細は末尾「Phase 2: proxy / equiv / gate」を参照。実 GPU・実モデルでの検証は Phase 3。

## Phase 1: `prefixanvil analyze`

リクエストログ（OpenAI 互換 JSONL）を再生し、

- (i) 実際に共有できた prefill 量（正規化なし）
- (ii) 意味を変えない決定的正規化（L1）を適用していれば共有できた量
- (iii) 参考値: 全リクエストが理想的な安定 prefix を持った場合の **上限**（達成可能値ではない）

を並べ、prefix が壊れた原因を F1–F8 に分類してトークン損失の大きい順に報告します。

> **数字の位置づけ:** 本ツールのシミュレータは vLLM V1 Automatic Prefix Caching の
> **仕様に沿った近似** であり、vLLM の再現ではありません。実機（実 GPU / 実 vLLM）との誤差検証は
> Phase 3 の仕事で、Phase 1 では未実施です。前提とパラメータは全てレポートに出力されます。

## クイックスタート

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # GPU なし・ネットワークなしで全テストが通る
.venv/bin/python demo.py               # fixture で gen → analyze を通す（out/demo/ に出力）
```

CLI（`pip install -e .` すると `prefixanvil` コマンド、未インストールなら `PYTHONPATH=src python -m prefixanvil`）:

```bash
prefixanvil gen --config fixtures/configs/demo.json --out trace.jsonl          # trace.labels.jsonl も出力
prefixanvil analyze --trace trace.jsonl --out report.md                        # report.json も出力
prefixanvil analyze --trace trace.jsonl --out report.md --labels trace.labels.jsonl   # 分類器の評価付き
prefixanvil analyze --trace prod.jsonl --out report.md --tokenizer /models/llama-3.1-8b-instruct
```

主なオプション: `--block-size 16`, `--cache-blocks 8192`, `--hash-model-name`,
`--prefill-tps 10000`（GPU 秒換算に使う **仮定** の prefill スループット）, `--order file|timestamp`,
`--chat-template FILE`, `--allow-download`（既定オフ）。

**tokenizer の既定値は同梱の fixture tokenizer です。** 実モデルではないためトークン数は参考になりません。
実ログには必ず `--tokenizer` で実モデルの tokenizer ディレクトリを指定してください（レポートにも警告が出ます）。

## 入力フォーマット

1 行 1 リクエストの JSONL。

| フィールド | 必須 | 内容 |
|---|---|---|
| `model` | ✓ | 文字列 |
| `messages` | ✓ | OpenAI 互換。`content` は文字列 / テキストパート配列 / （tool_calls 付き assistant のみ）null |
| `tools` | | OpenAI 互換 tools 配列 |
| `session_id` | | 文字列 or 整数。空文字は「なし」扱い |
| `timestamp` | | 数値（epoch 秒）か ISO 8601 文字列 |
| `request_id` | | 任意の識別子（ラベル照合に使用） |

検証で落ちた行は黙って捨てず、理由別件数と行番号サンプルをレポートに出します
（`invalid_json`, `missing_model`, `empty_messages`, `invalid_role`, `missing_content`, `invalid_tools`, `invalid_timestamp` …）。
テンプレート適用で失敗した行も `render_failures` として別集計します。

## アーキテクチャ

```
src/prefixanvil/
  ingest.py      JSONL 読み込みと検証（不正行は理由付きカウント）
  render.py      chat template 適用 → トークン列化。テンプレの出所とハッシュ版を記録
  simulator.py   ブロック・シミュレータ（APC 近似、LRU 退避）
  canonical.py   L1 正規化（決定的変換のみ）と変換ログ
  classify.py    first divergence の検出と F1–F8 分類、正解ラベルでの評価
  pipeline.py    analyze の本体（i / ii / iii の再生と集計）
  report.py      JSON + Markdown レポート
  agentload.py   合成エージェント・トレース生成器（正解ラベル付き）
  core.py        [P2] ステートレスなコア API（バージョン付き JSON 入出力、fail-open）
  relocate.py    [P2] L2 変換（揮発情報だけの行を system 末尾へ移設）
  proxy.py       [P2] prefixanvil proxy（stdlib http.server / http.client）
  mock_upstream.py [P2] 決定的モック upstream（テスト・equiv 用）
  equiv.py       [P2] prefixanvil equiv（等価性ゲート）
  gate.py        [P2] prefixanvil gate（CI 回帰ゲート）
  cli.py         prefixanvil gen / analyze / proxy / equiv / gate（argparse）
fixtures/
  tokenizer/     テスト用小型 byte-level BPE（2k 語彙）と ChatML 風テンプレート、再生成スクリプト
  configs/       AgentLoad 設定（F 単独注入 ×8、clean、demo）
  traces/        上記から生成したトレースと正解ラベル、不正行入りの malformed.jsonl
  equiv/         [P2] 等価性ゲートのケース（prefixanvil.equiv-case/v1）
  gate/          [P2] demo ベースライン（prefixanvil.gate-baseline/v1）と回帰用設定
.github/workflows/prefixanvil-gate.yml   [P2] pytest → prefixanvil gen → prefixanvil gate
tests/           pytest（130 件）
demo.py
```

## シミュレータのモデル（vLLM V1 APC との対応と差分）

| 項目 | 本シミュレータ | vLLM V1 | 備考 |
|---|---|---|---|
| ブロックサイズ | 既定 16、設定可 | GPU 既定 16 | |
| キャッシュ対象 | 完全ブロックのみ | 同じ | 末尾の部分ブロックは再利用されない |
| ハッシュ | SHA-256(親ハッシュ ‖ ブロックのトークン列 ‖ 任意で model 名) | hash(親, トークン列, extra_keys) | vLLM の extra_keys は LoRA / マルチモーダル / cache_salt。model 名は含まない（1 エンジン 1 モデル）ので本ツールでは任意オプション |
| 照合 | 先頭から連鎖、最初のミスで停止 | 同じ | |
| 最終トークン | 常に再計算（ヒット上限 = ⌊(n−1)/B⌋ ブロック） | 同じ（logits 取得のため） | |
| 退避 | 固定容量の物理ブロックプール、空きブロックの LRU キュー先頭から確保 | 同じ方式の free block queue | |
| 解放順 | リクエスト終了時に逆順で LRU 末尾へ（末尾ブロックから先に退避） | 同じ | |
| 並行性 | **なし**: 1 件ずつ逐次処理 | 並行リクエストがブロックを参照中は退避不可、バッチ内共有あり | 高負荷時の実ヒット率は本ツールと乖離し得る |
| decode トークン | **キャッシュに積まない** | 生成トークンの完全ブロックもキャッシュされる | 次ターンが前ターンの生成を含む場合、実機の方がヒットし得る（再レンダリングが生成トークン列と一致する場合のみ） |
| その他 | 未対応 | sliding window / hybrid モデルの複数 KV グループ、prompt_logprobs 時の無効化など | |

決定性: ハッシュは SHA-256 で `PYTHONHASHSEED` 等に依存しません（テストでゴールデン値を固定）。

> Phase 2 注記: シミュレータ自体は変更なし。条件 (ii) の L1 は session-aware なツール順序（下記 L1 節）で
> 再生するようになりました（`AnalyzeOptions(session_aware_l1=False)` で Phase 1 のステートレスソート）。

## 3 つの条件

- **(i) 実測相当**: ログをそのまま再生。
- **(ii) L1 適用後**: 各リクエストに L1 正規化を適用して同じシミュレータで再生。
- **(iii) 参考上限**: L1 後のトークン長で、各リクエストが「同一セッションの最長の先行プロンプトを完全に延長」
  「セッション先頭は同一モデルの静的 prefix（system + tools）を共有」できたと仮定し、
  **容量無限・退避なし** で計算。さらに (i)(ii) の実ヒットを下回らないよう max を取る。
  **到達可能性を意味しない上限** です。

GPU 秒 = 再計算 prefill tokens ÷ 仮定スループット（既定 10,000 tok/s）。線形の概算で、
attention の二次項・バッチング・GPU 種別は考慮しません。

## L1 正規化（意味を変えない決定的変換のみ）

| ルール ID | 内容 | fail-open の条件 |
|---|---|---|
| `L1.tools.sort_keys` | tools の再帰的キーソート（配列の順序は不変） | — |
| `L1.tools.sort_by_name` | tools 配列を関数名でソート（セッション文脈なしの場合のみ = Phase 1 の what-if） | 名前のないツール、名前重複 |
| `L1.tools.session_stable_order` | [P2] セッション内でツール順序を安定化（下記） | 同上 |
| `L1.json_string.normalize` | `tool_calls[].function.arguments` と tool ロールの content のうち JSON object/array として解釈できる文字列を、キーソート・固定セパレータで再シリアライズ | JSON でない / スカラー / 想定外の形 |
| `L1.system.whitespace` | system/developer の文字列 content: 行末空白除去、行中の連続空白を 1 つに（行頭インデントは保持）、3 連以上の改行を 2 つに、末尾改行除去。[P2] ``` / ~~~ フェンス内と 4 スペース/タブでインデントされた行（コード）は変更しない | content が文字列でない |

適用したルール ID と変換前後の SHA-256（先頭 16 桁）を全て記録します。入力は変更しません。冪等であることをテストしています。

`sort_by_name` は「ツール集合が同一の場合のみ」という指示を **ステートレスなソート** として実装しています
（同じ集合なら必ず同じ順序になる）。集合が変わる場合もソートされるため、追加/削除（F3）では divergence 位置が
前に動いて損失が増えることがあり、デモでも観測されています（下記「論点」参照）。

**[Phase 2] session-aware なツール順序.** `canonicalize(req, ctx)` は呼び出し側が持つ `SessionContext`
（`prefixanvil.session-context/v1`: 直前リクエストのツール名多重集合と、直前に *出力した* 順序）を受け取ります。

- 文脈なし: Phase 1 のステートレスな名前ソート（ログに `tool_sort:stateless_no_session_context`）。
- セッション先頭: クライアントの順序のまま（先頭をソートすると後続の追加/削除の分岐位置が動くため）。
- 多重集合が不変: 直前に出力した順序を再利用（名前ソートは `session_order_pinned` で skip）。クライアントの並び替え（F3 順序のみ）が消える。
- 多重集合が変化: 名前ソートは `tool_set_changed` で skip。残ったツールは直前の出力順、新規ツールは末尾。
  クライアントが並び替えていなければ結果はクライアントの順序そのもの。

demo の F3 損失: raw 784 → Phase 1 ステートレス 1,280 → session-aware 816 tokens。
順序は raw と同一で、残る +32 は `L1.tools.sort_keys` がトークン長を変えることによるブロック境界のずれです
（F3 単独フィクスチャでは 3,280 → 2,832）。

## 原因分類（F1–F8）

各リクエスト C について参照 P を選び（同一セッションの直前 / セッション先頭なら直前セッションの先頭（同モデル）/
session_id なしなら直近 16 件から最長共通 prefix のもの）、**最初の食い違い** だけを分析します
（prefix cache はそこで連鎖が切れるため、それ以降の差分は損失に影響しない）。

1. テンプレート版が違う → F6
2. レンダリング文字列の最初の食い違い位置を求め、tools と messages[j] のどちらが原因かを
   **反実仮想レンダリング**（C の tools を P のものに差し替えて食い違いが後ろへ動くか）で判定。テンプレート非依存。
3. 要素ごとのルール（先に当たったもの）:
   - tools: 集合差 → F3 / 順序のみ → F3 / JSON 等価 → F2 / 空白のみ → F6 / 揮発値 → F1 / その他 → unknown
   - system: JSON 等価 → F2、空白のみ → F6、揮発値パターン（ISO 日時、`26 Jul 2024`、曜日付き日付、UUID、epoch、hex ID、`run-000123` 等）→ F1、
     節（空行区切り、無ければ行）の追加/削除 → F5（長文 ≥ 512 文字で後ろに安定部があれば F7）、
     長文ブロックの差し替え → F7、短い固有名詞的スパンのみの置換 → F4、それ以外 → **unknown**
   - 過去ターン（同一セッション）: JSON 等価 → F2、空白のみ → F6、揮発値 → F1、それ以外 → F8
   - 構造が同一なのにレンダリングが違う（テンプレートが履歴を再レンダリング）→ F6
   - セッション間で非 system メッセージが最初に違う → `none`（別会話の自然な分岐）

損失 tokens（原因別内訳）= ペア単位で「安定 prefix なら再利用できたブロック量 − 実際の共通 prefix のブロック量」。
キャッシュ退避の影響は含まないため、(i) と (ii) の差とは一致しません。

### 分類器の評価

`prefixanvil analyze --labels` で AgentLoad の正解ラベルと照合し、F 別の適合率/再現率と混同行列を出します。

実測（12 設定 × 30 セッションのスイープ、system 長 150–1500 tokens、tools 0–12、retry 10%、各 F 10%）:
**全 F で再現率・適合率 100%**（F1 223/223, F2 169/169, F3 204/204, F4 30/30, F5 211/211, F6 196/196, F7 204/204, F8 151/151）。

**この 100% は実データでの精度を意味しません。** AgentLoad は「参照との差分がちょうど 1 イベント」になるよう
設計されており、生成器と分類器を同じ作者が書いているため、主に内部整合性の確認です。過剰適合を避けるため、
テストの下限は再現率・適合率とも **0.8** に緩く固定し、加えて生成器と異なる形の手書きケース
（Llama 形式の日付、日本語の人名、system 内 JSON、CRLF、行単位の節トグル等）で検証しています。
手書きケースで判明した既知の限界: 非 JSON の key/value 行の並び替え（YAML 風）は F2 にならず `unknown`。

## AgentLoad（合成トレース生成器）

- パラメータ: `sessions`, `turns`, `system_tokens`（概算、`chars_per_token`=4）, `tools`,
  `tool_burst_min/max`（1 ターン内の連続ツール呼び出し数）, `retry_rate`（同一内容の再送）, `faults`（F 別注入確率）, `seed`
- 1 リクエストあたり最大 1 イベントを注入（参照との差分が一意になるように）。各スコープで確率の和 ≤ 1。
- F4（ユーザー別補間）はセッション間のみ、F8（履歴書き換え）はセッション内のみ。他はどちらでも発生。
- イベントでバイト列が変わらなかった場合はラベルを出さない。
- 正解ラベル `<out>.labels.jsonl`: `request_index`, `request_id`, `fault`, `scope`, `reference_request_id`,
  `location`（`element`: `system` / `tools` / `messages[i]`、`section`、`detail`）。
- シード固定で出力バイト列が完全一致（`fixtures/traces` の再現性をテストで確認）。

## 前提と制約（要約）

- 再生順は既定でファイル順（`--order timestamp` で安定ソート）。
- テキストパート以外の content パートは `<type>` プレースホルダとしてレンダリング（マルチモーダルのハッシュは対象外）。
- 動作確認環境: Python 3.12.13, transformers 5.22.1, tokenizers 0.23.3, Jinja2 3.1.6。torch 不要。
- CLI は argparse（標準ライブラリで足り、依存を増やさないため）。

## Phase 2: proxy / equiv / gate

### コア API（SaaS 化の継ぎ目）

`prefixanvil.core.transform(payload)` は状態を持たない決定的な純関数で、入出力はバージョン付き JSON です。

```json
{"schema": "prefixanvil.transform.request/v1", "mode": "L0|L1|L2",
 "request": {"model": "...", "messages": [...], "tools": [...]},
 "session_context": {"schema": "prefixanvil.session-context/v1", ...} | null}
→ {"schema": "prefixanvil.transform.result/v1", "request": {...}, "changed": true,
   "applied": [{"rule", "path", "before_sha256", "after_sha256"}], "skipped": [...], "notes": [...],
   "would_apply": [...], "session_context": {...}, "before_sha256": "...", "after_sha256": "..."}
```

セッション状態は呼び出し側が保持し、毎回明示的に渡します。想定外の形・内部エラーは元のリクエストを
そのまま返し、`skipped` に `{"rule": "*", "reason": ...}` を記録します（fail-open）。
`core.Observer.observe` は L0 用に analyzer の分類器を 1 ペアに適用します（`prefixanvil.observe.*/v1`）。

### `prefixanvil proxy`

標準ライブラリのみ（`http.server` + `http.client`）。追加の実行時依存はありません。

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `PREFIXANVIL_UPSTREAM_BASE_URL` | （必須） | upstream のベース URL（例 `http://127.0.0.1:8000`） |
| `PREFIXANVIL_PORT` / `PREFIXANVIL_HOST` | `8080` / `127.0.0.1` | 待ち受け |
| `PREFIXANVIL_MODE` | `L0` | `L0` 観測（バイト同一で転送し、分類と「L1 なら何をしたか」を記録）/ `L1` 正規化して転送 / `L2` |
| `PREFIXANVIL_L2_ENABLE` | なし | `L2` には `1` が必須 |
| `PREFIXANVIL_EQUIV_REPORT` | `./equivalence-report.json` | L2 起動条件: 現在のビルドで合格した等価性レポート。無い・不合格・別ビルドなら起動拒否（exit 2） |
| `PREFIXANVIL_LOG` | なし | リクエスト毎の変換ログ（JSONL, `prefixanvil.proxy-log/v1`） |
| `PREFIXANVIL_TOKENIZER` | fixture | L0 の分類に使う tokenizer ディレクトリ |
| `PREFIXANVIL_MAX_SESSIONS` | `10000` | セッション文脈の LRU 上限 |

- 変換対象は `POST /v1/chat/completions` のみ。`GET /health`、それ以外のパスはそのまま中継。
- セッション識別: ヘッダ `X-PrefixAnvil-Session-Id` → body の `session_id` → `metadata.session_id`。無ければ L1 はステートレスソート。
  body の `session_id` は削除せず転送します。
- `"stream": true` の応答はそのままチャンク中継（正規化はリクエストのみ）。upstream の 5xx もそのまま返します。
  upstream に接続できない場合は 502。
- L2 は L1 に加えて 1 変換のみ: system 内の「ラベル + 揮発値だけ」の行（`Session clock: 2026-…Z`、`Request id: <uuid>`、
  `Run: run-000123` 等）を元の順序のまま system 末尾へ移動。コードフェンス内は対象外。F1 検出器の強いパターンのみ使用
  （裸の数値・時刻は対象外）。
- ログ: mode、適用/skip したルールと理由、前後ハッシュ、upstream ステータス、レイテンシ。プロンプト本文・セッション ID
  の生値は記録しません（セッション ID は SHA-256 先頭 16 桁、分類 detail はツール名を除いたコードのみ）。
  ログは upstream 応答が揃った時点（ストリームはヘッダ受信時点）で書き込みます。

```bash
PREFIXANVIL_UPSTREAM_BASE_URL=http://127.0.0.1:8000 PREFIXANVIL_MODE=L1 PREFIXANVIL_LOG=prefixanvil.jsonl prefixanvil proxy
```

### `prefixanvil equiv`（等価性ゲート）

```bash
prefixanvil equiv --suite fixtures/equiv/ --out equivalence-report.json   # + equivalence-report.md
```

各ケースのリクエスト（セッション列）を raw と変換後（L1、ケースが指定すれば L2）でモック upstream に送り、
出力の完全一致率と、不一致時の最初の差分位置を出します。ケースごとに `expect` が `match` / `allowed-diff`
（理由必須）で、想定外の不一致が 1 件でもあれば exit 1。レポートは `prefixanvil.equivalence-report/v1`
で、変換コードのビルド指紋（`canonical/relocate/core/classify/ingest` のソース + バージョンの SHA-256）を含みます。

**モック upstream（`prefixanvil.mock-upstream/v1`）は実モデルではありません。** 出力は宣言された等価モデル上の
意味内容の決定的関数です。非意味的と宣言しているもの: JSON のキー順・空白（ツール定義、arguments を 1 段だけ解釈、
JSON の tool 結果）、tools 配列の順序、system のコード外の行末空白・行中連続空白・3 連以上の改行・末尾改行、
`stream` などの転送フィールド。それ以外（インデント、フェンス内、行順、ユーザー/アシスタント本文、メッセージ順、
tool_call 順、非 JSON の tool 結果、サンプリングパラメータ）は意味的として扱います。
したがってこのゲートは「変換が宣言した等価性の範囲に収まっているか」を確かめるもので、
**実モデルの出力が変わらないことの証明ではありません**（Phase 3 で実モデルのサンプリング比較が必要）。
L2 の行移動はモック上は意味的変化なので `allowed-diff` として報告されます。

### `prefixanvil gate`（CI 回帰ゲート）

```bash
prefixanvil gate --trace out/ci/demo.jsonl --baseline fixtures/gate/demo-baseline.json \
           [--max-hitrate-drop 0.01] [--max-prefill-growth 0.02] [--write-baseline]
```

analyzer を再実行し、条件 (i) と (ii) それぞれについて、トークン加重ヒット率の低下（絶対値、0.01 = 1 ポイント）
または再計算 prefill tokens の増加（相対、0.02 = 2%）が閾値を超えたら exit 1。差分表は常に出力します。
ベースライン（`prefixanvil.gate-baseline/v1`）は `--write-baseline` を付けたときだけ作成/更新されます。
analyzer 設定（ブロックサイズ、容量、テンプレ版など）がベースラインと違う場合は比較不能として exit 2。
prefill tokens は絶対量の比較なので、同じワークロード（同じ設定から再生成したトレース）同士でのみ意味を持ちます。
トレースのバイト列がベースラインと違う場合は注記を出します。

`.github/workflows/prefixanvil-gate.yml` は push / PR で install → `pytest` → `prefixanvil gen`（demo 設定）→ `prefixanvil gate`
を実行します（ネットワークを使うのは依存のインストールのみ）。

## Phase 3: 計測ハーネス（`bench/`）

実 vLLM（無料枠の Kaggle / Colab T4）で測るためのハーネス。GPU 時間をコード書きに使わないよう、
すべてフェイクサーバでオフライン検証済み（`tests/test_bench.py`, `tests/test_free_gpu.py`）。

| モジュール | 内容 |
|---|---|
| `bench/replay.py` | AgentLoad トレースをセッション順を守って再生（並行 1 = キャッシュ会計の主計測、並行 8 = TTFT 参考値）。temperature 0、`max_tokens` 固定、TTFT・総時間・`cached_tokens` を記録 |
| `bench/metrics.py` | `/metrics` の検索ベースの名前解決（`vllm:prefix_cache_hits/queries`、旧名 `gpu_prefix_cache_*`、`request_prefill_kv_computed_tokens`、`cache_config_info`）。見つからなければ生テキストを保存 |
| `bench/vllm_ctl.py` | `POST /reset_prefix_cache`（`VLLM_SERVER_DEV_MODE=1` 時のみ存在）とリセットの挙動検証 |
| `bench/runner.py` | 条件ごとに リセット → ウォームアップ（破棄）→ リセット → 本計測 ×2。パス単位で保存・再開。予算超過時は縮退して記録 |
| `bench/fidelity.py` | 実トークナイザ・実ブロックサイズ・実 KV 容量でシミュレータを再生し実測と比較（クライアントのバイト列 / vLLM のサーバ側正規化 の 2 通り） |
| `bench/equiv_sample.py` | 実モデル等価性（raw 対 raw の対照、L1、L2、キャッシュ有無）。毎回コールドキャッシュ |
| `bench/free_gpu/` | `run_all.py`（1 回で完走・再開可能）、`run_dynamo.py`（Step 2, 90 分タイムボックス）、ソース埋め込み済みノートブック、`RUNBOOK.md` |

vLLM 0.33.0 のソースで確認した点（シミュレータとの差に効くもの）: tools は `model_dump()` で再シリアライズされてから
テンプレートに渡る（クライアントのキー順は `parameters` の中だけ残る）、`tool_calls[].function.arguments` 文字列は
dict にパースされる、`block_size` の既定は attention backend に応じて解決（既定値 16）。
`prefixanvil.render.Renderer(tok, "vllm")` でこの正規化を再現できます（既定は従来どおりクライアントのバイト列）。

### Phase 3 実測サマリ

Kaggle 2 × Tesla T4（無料枠）, Qwen2.5-7B-Instruct-AWQ（`@b250375…`）, vLLM 0.33.0, AgentLoad 合成トレース。
全数値は `BENCH.md`（`bench/make_bench_md.py` が結果 JSON から機械生成）が出典。**TTFT / GPU 秒は T4 参考値**で
H100 を代表しません。ヒット率・prefill は GPU 世代に依存しません。本計測 run `t4-20261111-0408`（全 11 ステージ ok、
予算カットなし、全条件 0 エラー）。

| 指標（`bench_main`, 並行 1, トークン加重） | 実測 | 事前登録（vLLM レンダリング） |
|---|---:|---:|
| ヒット率 raw | 74.71% | 74.63% |
| ヒット率 L1 正規化 | 81.79% | 81.82% |
| ヒット率 L2 移設（参考） | 83.61% | — |

- 再計算 prefill tokens: raw 241,640 → L1 172,268（**−28.7%**）→ L2 155,068（**−35.8%**）。
- シミュレータ忠実度（vLLM レンダリング）: 実測との差は 5 条件で −0.08 / +0.03 / −0.15 / −0.15 / −0.16 pt、
  プロンプトトークン総数はリクエスト単位で一致。
- セッション跨ぎ（`bench_xsession`）: セッション固定 86.69% / ステートレス名前ソート 95.31%（どちらも実測。
  デプロイ方針は保守的なセッション固定で、ステートレスを「勝ち」としては提示しない）。
- 実モデル出力サンプリング: L1 vs raw 10/18（変更分のみ 6/14）, L2 vs raw 4/6（1/3）, raw 対 raw 対照 18/18,
  ウォーム対コールドでもキャッシュ状態だけで 1/18 変化。**L1/L2 はプロンプト変換で、生成テキストの一致は保証しない**
  （等価性は `fixtures/equiv/` の決定的スイートが CI で担保）。
- NVIDIA Dynamo 試行（別 run `t4-20261110-2339` / vLLM 0.31.0 / ai-dynamo 1.6.2 / 2×T4, 150 リクエスト, 0 エラー、
  本計測とは別トレース・別条件）: worker 合算ヒット率 round-robin raw 60.4% / KV ルーティング raw 60.2% →
  KV ルーティング + L1 で 65.2%。
