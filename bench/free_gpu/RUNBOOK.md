# PrefixAnvil Phase 3 — 無料枠 GPU 実行手順書（依頼者向け）

この手順でやることは「ノートブックを 1 つアップロードして、すべて実行し、最後に zip を 2 つダウンロードする」だけです。
**課金は発生しません**（無料枠のみ。クレジットカード登録を求められたら、その時点で中止して連絡してください）。
アカウント作成・ログイン・電話番号認証はご本人の操作です。こちらからは何も入力しません。

## 0. 用意するもの

- `bench/free_gpu/prefixanvil_bench.ipynb`（このリポジトリにあるファイル。ソース一式が埋め込み済みなので、
  他にアップロードするものはありません）
- Kaggle アカウント（主）または Google アカウント（Colab、予備）

## 1. Kaggle（推奨）

1. https://www.kaggle.com/code → **New Notebook**。
2. メニュー **File → Import Notebook** → `prefixanvil_bench.ipynb` をアップロード。
3. 右パネル **Settings**:
   - **Accelerator**: `GPU T4 x2`（T4 x2 を選ぶと Step 2 の Dynamo 試行も自動で行います。選べない場合は `GPU T4`。
     P100 は選ばないでください）
   - **Internet**: `On`（Kaggle では通常、電話番号認証済みアカウントでないと On にできません）
   - **Persistence** は不要です。
4. **まず smoke 実行（推奨、約 30〜45 分）**: 2 番目のセルの `PROFILE = "full"` を `PROFILE = "smoke"` に変えて
   **Run All**（画面を開いたまま）。最後のセルに `prefixanvil_results_*.zip` のリンクが出れば成功です。
   この zip もダウンロードしておいてください（インストール・モデル・サーバ・全工程が動くかの確認用）。
5. **本計測**: `PROFILE = "full"` に戻し、右上 **Save Version → Save & Run All (Commit)** を選んで保存。
   ブラウザを閉じても裏で最大 12 時間走ります。
6. 終わったら（Notebook の **Versions** で Complete 表示）、そのバージョンの **Output** タブから
   `pf/prefixanvil_results_<run-id>.zip` と（あれば）`pf/prefixanvil_dynamo_<run-id>.zip` をダウンロード。

**所要時間の目安（T4 の実速度は未確認のため幅があります）**

| 工程 | 目安 |
|---|---|
| vLLM インストール | 10〜20 分 |
| モデル DL（Qwen2.5-7B-Instruct-AWQ, 約 5.5 GB） | 3〜10 分 |
| vLLM 起動 | 3〜8 分 |
| smoke（12 リクエスト × 2） | 〜5 分 |
| 本計測（主計測 3 条件 × (ウォームアップ + 2 巡)、セッション跨ぎ 3 条件、並行 8 の 2 条件） | 4〜6 時間 |
| 実モデル等価性サンプリング | 10〜20 分 |
| Step 2 Dynamo（T4 x2 の時のみ、上限 90 分） | 〜90 分 |

Kaggle の週あたり GPU 割当（多くの場合 30 時間前後。Settings で残量を確認できます）のうち、
smoke + 本計測 + Dynamo で **およそ 6〜9 時間** を使う見込みです。本計測は 9.5 時間の予算を超えそうになると、
自動で「2 巡 → 1 巡」「ウォームアップ短縮」「任意条件の省略」の順に縮め、その判断を結果に記録します。

## 2. Google Colab（予備）

1. https://colab.research.google.com → **ファイル → ノートブックをアップロード** → `prefixanvil_bench.ipynb`。
2. **ランタイム → ランタイムのタイプを変更 → T4 GPU**。
3. **ランタイム → すべてのセルを実行**。**タブを開いたまま** にしてください（無料枠はアイドルや長時間で切断されます）。
4. 最後のセルで zip が自動ダウンロードされます。切断された場合 `/content` は消えるので、
   Colab では smoke 実行と、時間内に収まる範囲の本計測（`BUDGET_HOURS` を小さく）を推奨します。
   Colab は GPU 1 枚なので Step 2（Dynamo）は実行されません（「未実施」として記録されます）。

## 3. 結果の戻し方

ダウンロードした zip（`prefixanvil_results_*.zip`、あれば `prefixanvil_dynamo_*.zip`、smoke の zip も）を

```
~/Desktop/Nvidia/PrefixAnvil/bench/results/incoming/
```

に置いて、「置いた」とだけ伝えてください。展開・検証・BENCH.md への反映はこちらで行います。
zip にはモデル出力の本文は「等価性比較で不一致だったペア」しか含まれず、プロンプトは合成データ（AgentLoad）のみです。

## 4. よくある失敗と対処

| 症状（ログ末尾） | 対処 |
|---|---|
| `No GPU attached` | Settings の Accelerator を GPU T4 x2（Colab は T4 GPU）に |
| `Internet is off` | Settings で Internet を On（Kaggle は電話番号認証が必要な場合あり） |
| `vLLM install failed for all pins` | そのまま zip を戻してください（`install.log` に原因が入っています）。こちらでピンを調整します |
| `STOP: ... license is ...` | モデルのライセンス表記が想定と違うため停止しています。zip を戻してください（判断はこちらで依頼者に確認） |
| `vLLM did not become healthy`（メモリ不足など） | 設定セルの `EXTRA_ARGS = ["--max-model-len", "6144", "--gpu-mem-util", "0.85"]` にして再実行 |
| 12 時間で途中終了（Kaggle） | 新しいバージョンで **Add Input → 前バージョンの Notebook Output** を追加し、`RUN_ID` を前回の run id、`RESUME_FROM` を `/kaggle/input/<追加した名前>/pf/results/<run-id>` にして再実行。完了済みの条件は飛ばされます |
| Colab が切断 | 同じ `RUN_ID` で再実行しても `/content` が消えていれば最初からになります。smoke の結果だけでも戻してください |

## 5. このノートブックがやること（透明性のため）

- `pip install uv` → `/tmp/pfvenv` に Python 3.12 の仮想環境 → vLLM をピン留めでインストール
  （ドライバの CUDA が 13 以上なら `vllm==0.33.0`、未満なら CUDA 12.9 ビルドの `vllm==0.32.0+cu129`。どちらを使ったか記録）
- Hugging Face から `Qwen/Qwen2.5-7B-Instruct-AWQ`（Apache-2.0）を、その時点のコミット SHA を固定してダウンロード。
  この GPU で AWQ モデルを起動できなかった場合に限り、理由を記録したうえで `Qwen/Qwen2.5-1.5B-Instruct`
  （Apache-2.0）に 1 回だけ自動で切り替えます（切り替えたことは結果と BENCH.md に明記されます）
- `vllm serve`（prefix caching 有効、`VLLM_SERVER_DEV_MODE=1` でキャッシュリセット用エンドポイントを有効化、localhost のみ）
- AgentLoad の合成トレースを再生（実ログは使いません）、`/metrics` のカウンタ差分で hit rate を計測
- Step 2 は `ai-dynamo[vllm]==1.6.2` を別の仮想環境に入れ、etcd/NATS 不要のファイル発見方式で 2 ワーカーを試行（90 分で打ち切り）
- 外部へのアップロード・投稿は一切しません。結果は `/kaggle/working`（または `/content`）の zip だけです。
