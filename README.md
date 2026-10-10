# ouj-notebook-converter

放送大学の PDF テキストを Markdown（および epub/PDF/txt）に変換する CLI ツール。

## 機能

- yomitoku による OCR（日本語対応レイアウト解析）
- Gemini API による OCR（yomitoku 未インストール環境向け）
- 数式変換（Pix2Text バックエンド）
- 章単位の Markdown 分割（`--split chapters`）
- 読み順推定（auto / left2right / right2left / top2bottom）

## インストール

### 前提

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) がインストール済みであること

### セットアップ

```bash
git clone https://github.com/mtane0412/ouj-notebook-converter.git
cd ouj-notebook-converter
uv sync
```

## 基本的な使い方

```bash
# PDF を Markdown に変換
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output

# ページ範囲を指定（1 ページ目、3〜5 ページ目、10 ページ目）
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --pages 1,3-5,10

# 章ごとにファイルを分割
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --split chapters
```

> **注意**: `--outdir (-o)` は必須オプションです。

## Gemini OCR バックエンド

yomitoku をインストールできない環境（依存関係の競合など）では、Gemini API を OCR バックエンドとして使用できる。

### セットアップ

プロジェクトルートに `.env` ファイルを作成し、API キーを設定する。

```
GEMINI_API_KEY=your_api_key_here
```

### 変換の実行

```bash
# .env に GEMINI_API_KEY を設定済みの場合
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --ocr-backend gemini

# API キーを直接指定する場合
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --ocr-backend gemini --gemini-api-key YOUR_KEY

# モデルを変更する場合（デフォルト: gemini-3.8-flash）
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --ocr-backend gemini --gemini-model gemini-3.5-flash
```

> **注意**: Gemini バックエンドは単語の位置情報（bbox）を返さないため、`--format pdf`（searchable PDF）とは併用できない。searchable PDF が必要な場合は `--ocr-backend yomitoku` を使用する。

## 数式変換

数式を含む PDF には `--math-backend` オプションを使用する。

| バックエンド | 概要 | 推奨シナリオ |
|---|---|---|
| `none` | 数式変換なし（デフォルト） | 数式のない PDF |
| `pix2text` | Pix2Text でページ全体から数式を検出・認識する | 数式を含む PDF |

### Pix2Text バックエンドのセットアップ

Pix2Text は yomitoku と依存関係が競合する可能性があるため、専用の venv で動かす。

#### 1. Pix2Text venv の作成

```bash
python3.11 -m venv ~/.venvs/pix2text
~/.venvs/pix2text/bin/pip install "pix2text[serve]"
```

#### 2. 変換の実行（サーバーは自動起動）

```bash
uv run ounc 数式入りPDF.pdf --outdir /tmp/output --math-backend pix2text
```

`--math-auto-start`（デフォルト: 有効）により、サーバーが未起動の場合は自動で起動する。
モデルのロードに 10〜30 秒かかる。

カスタム URL を使う場合:

```bash
uv run ounc 数式入りPDF.pdf --outdir /tmp/output \
    --math-backend pix2text \
    --pix2text-url http://localhost:9000
```

#### 3. サーバーを手動で起動する場合（オプション）

事前にサーバーを起動しておく場合は `--no-math-auto-start` を指定する。

```bash
# 別ターミナルでサーバーを起動
~/.venvs/pix2text/bin/python scripts/pix2text_server.py --port 8503

# 自動起動を無効にして変換
uv run ounc 数式入りPDF.pdf --outdir /tmp/output \
    --math-backend pix2text \
    --no-math-auto-start
```

## 人手修正ファイル

人が確認した OCR の誤読は、キャッシュの `raw.md` を書き換えず、修正ファイル（JSON）として別管理し、
`--corrections` で変換時に適用する。再 OCR（`--no-cache` や OCR 設定の変更）をしても修正は失われず、
いつ誰が何を直したかが記録として残る。修正ファイルは教材の本文を含むため、リポジトリ外
（例: 書籍フォルダ直下の `corrections.json`）に置く。

```bash
uv run ounc book.pdf -o out --corrections /path/to/corrections.json
```

### 形式（version 1）

```json
{
  "version": 1,
  "corrections": [
    {
      "page": 70,
      "before": "(\\sqrt[3]{a})^4 > (\\sqrt[3]{a})^3$ だから",
      "after": "(\\sqrt[4]{a})^4 > (\\sqrt[4]{a})^3$ だから",
      "reason": "4 乗根の添字の誤読",
      "reviewer": "mtane0412",
      "date": "2026-10-10"
    }
  ]
}
```

| キー | 必須 | 内容 |
|------|------|------|
| `page` | 必須 | PDF のページ番号（1 始まり。キャッシュの `page_NNNN` と同じ） |
| `before` | 必須 | 置換前の文字列。当該ページの OCR Markdown（`raw.md`）にちょうど 1 か所だけ現れること |
| `after` | 必須 | 置換後の文字列（`before` と同じは不可） |
| `reason` / `reviewer` / `date` | 任意 | 記録用。適用結果には影響しない |

- 同一ページの修正は配列の記載順に適用する。未知のキーは形式エラーになる
- 置換前が見つからない場合・複数一致する場合・`page` が総ページ数を超える場合は、
  どのページのどの修正かを示して変換を停止する（暗黙にスキップしない）。複数一致する場合は前後の文字を足して一意にする
- `--pages` の対象外のページの修正は適用されない（総ページ数の範囲内であれば検証もされない）

### 適用タイミング

`build_page_markdown` の中で、`raw.md` を読み込んだ直後（数式 overlay の適用・`normalize_ocr_markdown` より前）に適用する。
`before` / `after` は OCR が出力した `raw.md` 上の表記で書き、後処理の正規化規則の影響を受けない。
章検出に使う `raw_markdown` にも修正後のテキストが渡る。LLM 判定の結果もこの形式で書き出して適用する想定である。

## オプション一覧

```
Usage: ounc [OPTIONS] INPUT_PDF

Arguments:
  input_pdf  変換する PDF ファイルのパス

Options:
  -o, --outdir PATH                           出力先ディレクトリ（必須）
  -f, --format [md|epub|pdf|txt]              出力形式（複数指定可）[default: md]
  -d, --device TEXT                           推論デバイス: mps / cpu / cuda [default: mps]
      --dpi INTEGER                           PDF レンダリング DPI [default: 200]
      --pages TEXT                            処理するページ範囲 例: 1,3-5,10
      --cache-dir PATH                        キャッシュディレクトリ
      --no-cache                              キャッシュを無効化
      --combine/--no-combine                  全ページを 1 ファイルに結合 [default: combine]
      --reading-order [auto|...]              読み順推定モード [default: auto]
      --ignore-meta/--no-ignore-meta          ヘッダ/フッタを除外 [default: ignore-meta]
      --split [none|chapters]                 出力分割モード [default: none]
      --math-backend [none|pix2text]          数式変換バックエンド [default: none]
      --pix2text-url TEXT                     Pix2Text ラッパー URL [default: http://localhost:8503]
      --pix2text-venv PATH                    pix2text 用 venv のパス [env: OUC_PIX2TEXT_VENV]
                                              [default: ~/.venvs/pix2text]
      --math-auto-start/--no-math-auto-start  pix2text 時にサーバーを自動起動 [default: math-auto-start]
      --ocr-backend [yomitoku|gemini]         OCR バックエンド [default: yomitoku]
      --gemini-api-key TEXT                   Gemini API キー [env: GEMINI_API_KEY]
      --gemini-model TEXT                     Gemini モデル名 [default: gemini-3.8-flash]
      --corrections PATH                      人手修正ファイル（JSON）。変換時に適用する
  -v, --verbose / -q, --quiet
```

## 開発

### テスト実行

```bash
uv run pytest tests/unit -v
```

### 型チェック

```bash
uv run mypy src tests
```

### Lint / フォーマット

```bash
uv run ruff check src tests
uv run ruff format src tests
```

### 精度評価

人手で作成した正解 Markdown（評価セット）と OCR 出力を比較し、精度指標を算出する。
評価セットは教材の著作物を含むためリポジトリには置かず、場所を `--truth` で指定する。

```bash
# 初回のみ: KaTeX 描画検査用の Node.js 依存をインストール
(cd scripts/katex_check && npm install)

uv run python -m ouj_notebook_converter.evaluation \
  --truth /path/to/eval \
  --pred /path/to/output_or_cache \
  --json result.json
```

- `--truth`: `manifest.json`（`{"pages": [{"page": 70, "category": "数式中心"}]}`）と正解 `page_NNNN.md` を置いたディレクトリ
- `--pred`: `ounc --no-combine` の出力ディレクトリ、またはページキャッシュ（`page_NNNN/raw.md`）のディレクトリ
- 指標: 地の文の文字誤り率（CER）、数式の一致率（F1）と文字誤り率、見出し（レベル・テキスト）の一致率（F1）、KaTeX で描画できない数式の件数。ページ別・層別・全体で表示する

### 変換時の品質レポート（KaTeX 描画可否）

`--quality-report` を付けると、変換結果の全数式を KaTeX で検査し、描画できない数式を
「ページ / 数式 / エラー内容」で一覧にして出力先ディレクトリ（`--outdir`）に書き出す。

```bash
# 初回のみ: KaTeX 検査用の Node.js 依存をインストール
(cd scripts/katex_check && npm install)

ounc book.pdf -o out --ocr-backend gemini --quality-report
# → out/quality_report.json（機械可読）と out/quality_report.md（人が読む表）
```

- 既定では出力しない（オプトイン）。Node.js と `npm install` が前提であり、既定で有効にすると Node.js の無い環境の変換が失敗するため
- Node.js や katex が使えない場合は、OCR を始める前にエラー終了する（黙ってスキップしない）
- ページ番号は PDF の 1 始まり、数式番号はそのページ内の出現順（1 始まり）
- 後処理で `eqnarray*` → `aligned` への置き換えと `array` 列指定の `@{...}` の除去を自動で行う。
  `\cline` / `\multicolumn` / `\enclose` は KaTeX に同等の表記が無く、置き換えると筆算の意味が変わるため変換せず、レポートで検出する

### 等式の検算による誤読候補の検出（プロトタイプ）

OCR 結果の数式に含まれる等式・不等式の連鎖（`A = B > C` など）を sympy で検算し、
恒等的に成り立たない式（例: 4 乗根を 3 乗根と誤読した `(\sqrt[3]{a})^3 = (\sqrt[3]{a})^4`）を誤読候補として表示する。

```bash
uv sync --extra verify   # sympy と lark（LaTeX パーサー）をインストール

uv run python -m ouj_notebook_converter.detectors \
  --pred /path/to/output_or_cache \
  --truth /path/to/eval \
  --json result.json
```

- `--pred`: 精度評価と同じ。ディレクトリ内の全ページを検算する
- `--truth`（任意）: 評価セットのページについて、誤読候補を正解と照らし合わせて適合率・再現率を表示する
- 方程式・定義（片辺が文字 1 つや数値の等式、両辺の文字の組が異なる等式）、値によって成否が変わる不等式、
  「〜ではない」と否定された式は誤読候補にしない。日本語を含む項・解釈できない項・筆算や場合分けの環境はスキップし、件数を表示する

### 書籍内の整合性による誤字候補の検出（プロトタイプ）

1 冊の本の中での表記の一貫性を使い、地の文の誤字候補を API なしで検出する。
本全体で 1〜2 回しか出ない語が、3 回以上出る語と 1 文字だけ違う場合（例: 奥付の「服部正博」と本文の「隈部正博」）を候補にする。
追加の依存は不要（日本語の分かち書きは、漢字・カタカナの連続を語とみなして代用する）。

```bash
uv run python -m ouj_notebook_converter.detectors.consistency \
  --pred /path/to/output_or_cache \
  --truth /path/to/eval \
  --json result.json
```

- `--pred`: 精度評価と同じ。ディレクトリ内の全ページを対象にする
- `--truth`（任意）: 評価セットのページについて、正解との文字差分を真の誤読とみなして適合率・再現率を表示する
- `--skip-index`: 巻末索引が無い本で、索引語の検出を省く（索引ページを検出できない場合は、既定ではエラーにする）
- 検出器は 3 種類（結果は種類ごとに表示する）: 低頻度語（`rare_word`）、巻末索引の語彙に近い語（`index_near`）、
  ひらがな 1 文字違いの低頻度 n-gram（`rare_ngram`。誤検出が非常に多いため参考扱い）

### 式番号・例番号・脚注番号の連番検査（プロトタイプ）

式番号 (N.M)・ラベル（例・練習・コメント・図・定理・命題・系）・脚注番号が章ごとに連番であることを使い、
OCR が落とした式・ラベル・脚注や、誤読した番号を検出する。本文中の参照（「(2.9) より」）は定義と区別して数える。
`detectors/cli.py` には組み込まず、モジュール単位で実行する。

```bash
uv run python -m ouj_notebook_converter.detectors.numbering \
  --pred /path/to/output_or_cache \
  --json numbering.json
```

- `--pred`: 精度評価と同じ。ページキャッシュの `raw.md` には後処理（`normalize_ocr_markdown`）を適用して読み込む。
  `\tag{N.M}` と行末の「(N.M)」のどちらも定義とみなすので、後処理の前後どちらでも結果は変わらない
- 指摘の種類: `missing`（欠番。前後の定義があるページと参照元を表示）、`duplicate`（重複定義）、
  `order`（順序の逆転）、`undefined_reference`（最大の定義番号を超える番号への参照）
- 章の最大番号が欠落した場合は、その番号への参照が無い限り検出できない。Gemini 形式の Markdown（太字ラベル・`\tag`）を前提とし、
  yomitoku の出力は対象外

### yomitoku との差分による地の文の誤読候補の検出（プロトタイプ）

Gemini と yomitoku の OCR 結果を日本語部分（かな・漢字）だけで比較し、食い違い箇所を誤読候補として抽出する。
数式は yomitoku では構造が失われるため比較しない（Gemini の数式の位置は区切りとして扱う）。
どちらが正しいかは判定しない（後段の LLM 判定に渡す入力を作る）。

```bash
uv run python -m ouj_notebook_converter.detectors.cross_ocr_diff \
  --gemini /path/to/gemini_cache \
  --yomitoku /path/to/yomitoku_cache \
  --truth /path/to/eval \
  --json result.json
```

- `--gemini`: Gemini の出力（`--no-combine`）またはページキャッシュ（`page_NNNN/raw.md`）。精度評価の `--pred` と同じ
- `--yomitoku`: yomitoku のページキャッシュ。各ページの `page_NNNN/analysis.json`（段落の位置・図の領域・単語）を使う。無いページがあればエラーにする
- `--truth`（任意）: 評価セットのページについて、適合率・再現率を表示する。誤読は「Gemini の地の文と正解の地の文を文字単位で比べたときの、かな・漢字を含む食い違い」、
  誤読候補が誤読に当たるとは「候補の Gemini 側の位置が誤読の位置と 1 文字以内で重なる」こと。
  適合率 = 誤読に当たった候補数 / 候補数、再現率 = 候補が当たった誤読数 / 誤読数
- ノイズ除去（`--no-exclude-figures`・`--no-exclude-header`・`--no-normalize-variants`・`--min-run-length` で個別に無効化できる）
  - 図の領域にある段落と、ページ上部の帯（柱）にある段落は比較しない
  - 「一」と「ー」、小書きの仮名と通常の仮名は同じ文字とみなし、yomitoku 側が「ー」だけを読んだ食い違い（数式のマイナス・分数線）は候補にしない
  - Gemini の数式・変数の位置に yomitoku が 1〜2 文字の仮名・記号を読んだ食い違い、段落に取り込まれた図番号の「図」「表」は候補にしない
  - 置き換えの片側が 5 文字を超える食い違いは、読む順序の違いによる対応の取り違えとして候補にしない
  - Gemini に対応箇所が無い日本語の連続は、既定では候補にせず件数だけ数える（`--report-unmatched` で候補にする）

#### 後段の LLM 判定（#24）への入力形式

`--json` の出力は次の形式（`format_version: 1`）。`candidates` の各要素が 1 つの食い違い。

```json
{
  "format_version": 1,
  "options": {"min_run_length": 3, "exclude_figures": true, "exclude_header": true,
              "normalize_variants": true, "report_unmatched": false, "context_chars": 15},
  "page_count": 314,
  "stats": {"paragraphs": 3670, "runs": 10578, "runs_exact": 7618, "...": 0},
  "candidate_count": 77,
  "candidates": [
    {
      "page": 60,
      "kind": "gemini_only",
      "gemini": {"text": "制", "context_before": "かの数の並びが無", "context_after": "限に繰り返されて"},
      "yomitoku": {"text": "", "context_before": "かの数の並びが無", "context_after": "限に繰り返されて",
                   "bbox": [110, 802, 996, 1433], "bbox_source": "paragraph"}
    }
  ],
  "scores": {"pages": ["..."], "total": {"precision": 0.214, "recall": 1.0}}
}
```

- `page`: ページ番号（PDF の 1 始まり）。原本画像は `bbox` を切り出して LLM に見せられる
- `kind`: `replace`（両方に文字があり異なる）／`gemini_only`（Gemini にだけある）／`yomitoku_only`（yomitoku にだけある。Gemini の脱落の疑い）／`unmatched_run`（`--report-unmatched` のときだけ）
- `gemini.text`: Gemini 側の食い違い部分。`yomitoku_only` では空で、`context_before` と `context_after` の間が挿入位置になる。
  `context_*` は Gemini の地の文（空白と Markdown 記法を除き NFKC 正規化したもの）の前後 15 文字で、数式のあった位置は `〔数式〕` と書く
- `yomitoku.text`: yomitoku 側の食い違い部分（`gemini_only` では空）。`context_*` は同じ段落内の前後 15 文字
- `yomitoku.bbox`: `[x0, y0, x1, y1]`（200 DPI で描画した原本ページ画像の px 座標）。食い違い部分を含む単語（行）が 1 つに決まれば `bbox_source: "word"`、決まらなければ段落の座標で `"paragraph"`

### 人手確認用ページ（校正台）の生成

評価セットの正解 Markdown を、原本ページ画像と並べて人手で確認するための HTML を生成する。
左にページ一覧（層・確認状態・KaTeX エラー件数）、中央に原本画像（拡大縮小可）、右に
「正解（描画）」「比較対象（描画）」「差分」「Markdown」「照合記録」のタブ、下部に「確認OK／要修正＋メモ」がある。

```bash
# 初回のみ: KaTeX（0.18.9）を取得する。CSS・フォント・JS は生成時に index.html へ埋め込む
(cd scripts/katex_check && npm ci)
uv sync --extra gemini   # ページ画像の JPEG 化に Pillow を使う

uv run python -m ouj_notebook_converter.evaluation.review_board \
  --truth /path/to/eval \
  --pdf /path/to/原本.pdf \
  --out /path/to/experiments/issue-N/board \
  [--pred /path/to/draft_or_cache --pred-label "Gemini下書き"]
```

- 出力: `index.html`（KaTeX とデータを埋め込んだ 1 ファイル）、`data.json`（同じデータの複製）、`img/page_NNNN.jpg`（200 DPI グレースケール）。教材の画像・本文を含むためリポジトリには置かない
- `--pred`（任意）: 精度評価と同じ形式（`page_NNNN.md` または `page_NNNN/raw.md`）。指定すると比較対象の描画と「比較対象→正解」の差分タブが現れる。manifest の全ページ分が必要
- 数式の検出は変換時の後処理（`markdown_cleanup.MATH_SPAN`）の正規表現を `data.json` の `mathSpanPattern` に載せて使う。KaTeX で描画できない数式は赤く表示し、一覧にエラー件数を出す
- テンプレートは `src/ouj_notebook_converter/evaluation/templates/review_board.html`

確認結果の保存先と取り込み方:

- ローカルで `index.html` を開いた場合（db が無い）: ブラウザの localStorage に保存する。ヘッダーの「確認結果を JSON でダウンロード」で `reviews.json`（`{"reviews": [{"page", "category", "status", "memo", "updatedAt"}]}`）を書き出せる。`status` は `ok` または `fix`
- Artifact として公開した場合: `capabilities: {db: {}}` を宣言して公開する（`index.html` と `data.json`・`img/` を `files` で渡す）。確認結果は db の `reviews` コレクションに、ドキュメント ID `p0015`（ページ番号 4 桁）、本体 `{page, status, memo, updatedAt}` で保存される
- db の結果の取り込み: Claude が `ArtifactData` の `list`（collection=`reviews`）で読み戻し、上記と同じ形式の JSON にして保存する。以降の処理（#24・#28 など）はその JSON を入力にする

## アーキテクチャ概要

```
[CLI: ounc]
    │
    ▼
[ConvertConfig]
    │
    ▼
[run_pages] ── per page ──▶ analyze_page (yomitoku OCR) → analysis.json
    │                                                          │
    │  math_backend=pix2text ──────────────────────▶ math_detect
    │                                                    │
    │  ← MathOverlay (items / roles / originals) ────────┘
    │
    ▼
[build_page_markdown] → PageMarkdown
    │
    ▼
[assemble_markdown] → .md ファイル
```

- `math_detect`: Pix2Text でページ全体を解析し、yomitoku paragraph と IoU マッチして LaTeX 化
