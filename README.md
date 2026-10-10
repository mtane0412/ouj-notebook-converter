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

### 評価セットのページだけを条件を変えて再 OCR する（実験用）

解像度・Gemini モデル・プロンプトを変えたときの精度を比べるための実験用ハーネス。評価セット（`manifest.json`）のページだけを再 OCR し、精度評価にそのまま渡せる形式で出力する。

```bash
export GEMINI_API_KEY=...   # または --api-key

uv run python -m ouj_notebook_converter.evaluation.reocr \
  --truth /path/to/eval --pdf /path/to/原本.pdf \
  --dpi 300 --model gemini-3.8-flash \
  --out /path/to/experiments/issue-25/flash-300dpi \
  [--prompt-file prompt.txt] [--run-id run1]

# 出力をそのまま評価できる
uv run python -m ouj_notebook_converter.evaluation \
  --truth /path/to/eval --pred /path/to/experiments/issue-25/flash-300dpi
```

- 出力: `page_NNNN/raw.md`（OCR 結果）、`page_NNNN/usage.json`（処理時間・試行回数・トークン使用量）、`run.json`（条件と合計）
- `--run-id`: 繰り返し実行用。指定すると `<out>/<run-id>/` に出力する
- 中断しても同じコマンドで再開できる（完了済みページは API を呼ばない）。レート制限（HTTP 429/500/503）は待って再試行する
- ページキャッシュ（`ounc`）のディレクトリ名は、既定以外のモデル・DPI を指定すると `…gemini-<モデル名>-<DPI>dpi` のように分かれる（既定値は従来どおり `…gemini`）

#### プロンプトの比較と後処理の発火件数（#26）

Gemini のプロンプトを変えたときの効果は、同一条件でも CER が 1.5〜3.6% 動くため、各条件を 3 回以上実行して平均と最小〜最大で比べる（1 回ずつの比較では判定できない）。

```bash
# 条件ごとに run1〜run3 を再 OCR する
for run in run1 run2 run3; do
  uv run python -m ouj_notebook_converter.evaluation.reocr \
    --truth /path/to/eval --pdf /path/to/原本.pdf \
    --prompt-file experiments/prompts/gemini_ocr_v3.txt \
    --out /path/to/experiments/issue-26/v3 --run-id $run
  uv run python -m ouj_notebook_converter.evaluation --truth /path/to/eval --pred /path/to/experiments/issue-26/v3/$run
  # 後処理（normalize_ocr_markdown）の各ルールが生の出力に何回発火したかを数える
  uv run python -m ouj_notebook_converter.evaluation.cleanup_stats --pred /path/to/experiments/issue-26/v3/$run
done
```

- `cleanup_stats` の出力は JSON。`hfill` / `tag` / `tag_only_math` / `quad` / `eqnarray_star` / `array_column_spacing` / `section_heading_level` / `label_heading` は後処理で直せた件数（合計が `total`）。`unfixable_*` と `empty_math` は後処理で直せない破綻の件数
- `experiments/prompts/gemini_ocr_v1.txt`〜`v3.txt` は #26 で比較した改善案。いずれも既定プロンプトより総合精度が上がらなかったため `gemini.py` には採用していない（結果は issue #26 を参照）。評価セットの正解が現行プロンプトの出力を下書きに作られているため、現行プロンプトと書式が異なる出力は不利に評価される点に注意する

### 複数回 OCR の揺れによる不確実箇所の検出と多数決（プロトタイプ）

同じページを同じ条件で複数回 OCR し、実行間で食い違う箇所（地の文・数式）を「OCR が自信を持てない箇所」として抽出する。
どれが正しいかは判定しない（後段の LLM 判定 #24 に渡す入力を作る）。`--truth` を指定すると、揺れと実際の誤読の重なりと、多数決で確定した場合の指標の改善幅も測る。

```bash
# 1 回分の OCR は evaluation.reocr の --run-id で繰り返せる（OCR 回数に比例して費用がかかる）
uv run python -m ouj_notebook_converter.detectors.multi_run_variance \
  /path/to/run1 /path/to/run2 /path/to/run3 \
  --truth /path/to/eval --subset-size 2 --subset-size 3 \
  --json result.json
```

- 引数: 1 回分の OCR の出力（`--no-combine` の出力、またはページキャッシュ `page_NNNN/raw.md`）を 2 つ以上。最初の実行を基準にして候補を出す
- 整列: 基準の実行に他の実行を文字単位（数式は 1 つ単位）で整列し、食い違う範囲を集める。地の文は評価と同じ正規化（Markdown 記法・空白・全角半角を除く）、数式は `normalize_latex` で表示が同じ表記の違いを除いて比べる
- `--min-dissent N`: 基準の実行と異なる実行が N 回以上ある食い違いだけを候補にする（既定 1）
- `--subset-size K`（複数指定可）: 実行の組み合わせ（例: 5 回から 3 回を選ぶ全 10 通り）ごとに評価して合算する。既定は全実行
- 適合率・再現率の定義（各実行を順に基準にして件数を合算する）
  - 地の文: 誤読 = 基準の実行の地の文と正解の地の文の文字単位の食い違い（かな・漢字に限らない）。候補が誤読に当たる = 候補の範囲が誤読の範囲と 1 文字以内で重なる。適合率 = 当たった候補数 / 候補数、再現率 = 候補が当たった誤読数 / 誤読数
  - 数式: 数式 1 つを単位とする（等式の検算と同じ定義）。誤読 = 正解の数式と多重集合として一致しない数式。適合率 = 候補にした数式のうち誤読だった数 / 候補にした数式の数、再現率 = 同 / 誤読の数
- 多数決: ページごとに medoid（他の実行との差が最小の実行）を基準にし、食い違う範囲ごとに最も多い版を採用する（同数なら基準の版）。地の文は文字単位、数式は 1 つ単位で投票する。指標は評価（`evaluation`）と同じ定義の CER・数式F1・数式CER

#### 後段の LLM 判定（#24）への入力形式

`--json` の出力（`format_version: 1`）の `candidates` の各要素が 1 つの揺れ。

```json
{
  "format_version": 1,
  "run_dirs": ["/path/to/run1", "/path/to/run2", "/path/to/run3"],
  "options": {"min_dissent": 1, "context_chars": 15},
  "page_count": 33,
  "candidate_count": 12,
  "candidates": [
    {
      "page": 3,
      "granularity": "prose",
      "reference_run": 0,
      "run_count": 3,
      "dissent": 1,
      "texts": ["服", "隈", "隈"],
      "context_before": "著者は",
      "context_after": "部正博です。",
      "reference_span": [3, 4],
      "formula_indices": []
    }
  ],
  "scores": {"detection": {"3": {"1": {"prose": {"precision": 0.5, "recall": 0.8}, "formula": {"...": 0}}}},
             "vote": {"3": {"single_mean": {"cer": 0.03}, "medoid": {"...": 0}, "voted": {"...": 0}}}}
}
```

- `granularity`: `prose`（地の文）／`formula`（数式）
- `texts`: 実行ごとの文字列（`run_dirs` と同じ順）。数式は正規化前の LaTeX（複数なら改行区切り）。その実行に対応する文字・数式が無ければ空文字列
- `context_before` / `context_after`: 基準の実行の地の文での前後 15 文字。数式のあった位置は `〔数式〕`
- `dissent`: 基準の実行と文字列が異なる実行の数（`run_count - 1` に近いほど、どの実行も一致しない箇所）
- `formula_indices`: `formula` のみ。範囲に含まれる基準の実行の数式の番号（全数式を出現順に数えた 0 始まり）

### 誤読候補の LLM 判定（プロトタイプ）

各検出器（等式の検算・書籍内整合性の低頻度語・yomitoku との差分・KaTeX 描画不能）が挙げた候補を、
原本画像の該当部分・OCR の該当行・検出器の根拠と一緒に Gemini に渡し、「誤読か」「正しい表記」を
構造化出力（JSON）で判定させる。判定結果は修正ファイル（上の「人手修正ファイル」、`version: 1`）に書き出せる。
連番検査（`numbering`）は「無いもの」の報告で置換の対象となる文字列が無いため、判定の対象にしない。

```bash
uv sync --extra verify --extra gemini
export GEMINI_API_KEY=...

# 1. 各検出器の候補を集め、原本画像上の位置を推定する（候補ファイル candidates.json）
uv run python -m ouj_notebook_converter.judge collect \
  --pred /path/to/gemini_cache --yomitoku /path/to/yomitoku_cache --out candidates.json

# 2. 候補を原本画像と一緒に判定させる（判定結果 judgments.json、LLM に渡した画像は crops/ に保存）
uv run python -m ouj_notebook_converter.judge judge \
  --candidates candidates.json --pdf /path/to/原本.pdf --model gemini-3.1-pro-preview \
  --out judgments.json --crops-dir crops

# 3. 修正ファイル（自動適用 corrections.json・確認待ち corrections_pending.json・振り分け結果 decisions.json）
uv run python -m ouj_notebook_converter.judge export \
  --candidates candidates.json --judgments judgments.json --pred /path/to/gemini_cache --out-dir out

# 4. 評価セットで判定の正誤・修正適用後の指標・コストを測る
uv run python -m ouj_notebook_converter.judge evaluate \
  --truth /path/to/eval --pred /path/to/gemini_cache --candidates candidates.json \
  --judgments judgments.json --known-corrections /path/to/corrections.json --labels labels.json
```

- `collect`: `--source` で検出器を選ぶ（既定は `equation_check` `rare_word` `cross_ocr_diff` `katex_error`。`--yomitoku` が無ければ `cross_ocr_diff` は使わない）。
  検出器には後処理前の `raw.md` をそのまま渡す（修正ファイルの `before` は `raw.md` 上の表記で書くため）。等式の検算（約 4 分）が大半の時間を占める
- `judge`: `--image-mode crop`（既定。該当箇所の切り出し）と `page`（ページ全体を長辺 `--page-max-side` px に縮小）を選ぶ。API の失敗・応答不正は握りつぶさず、判定結果の `error` に記録して終了コード 1 にする
- `export`: `--judgments` を複数指定すると、全ての判定が同じ修正で一致したものだけを自動適用する（`--min-agree` で一致数を変えられる）。不一致や判断不能は確認待ちになる
- `evaluate`: `--judgments a.json+b.json` のように `+` でつなぐと、複数の判定の合意で評価する。`--labels` は `{候補 id: "misread" | "correct"}` の JSON で、
  評価セット内は既知の人手修正（`--known-corrections`）の置換前の文字列を抜粋に含む候補を真の誤読とし、手動ラベルで上書きする。評価セット外の候補にラベルを付けると別枠で集計する

#### 候補の位置の特定（Gemini 出力には bbox が無い）

1. `cross_ocr_diff` は yomitoku の単語（行）の `bbox` を持つ。ただし yomitoku 側の対応の取り違えで `bbox` が別の箇所を指すことがあるため、Gemini の行を yomitoku の段落に対応付けた結果（下の 2.）と食い違えば、対応付けた段落に置き換える
2. 日本語を 6 文字以上含む Gemini の行を、日本語部分の文字 2-gram の Dice 係数（0.5 以上）が最大の yomitoku の段落に対応付ける（`line_alignment`）
3. 数式だけの行は日本語が無く対応付けできないため、前後の「対応付けできた行」の段落に挟まれた帯とする（`band_between`）。片側しか無ければ、その段落から 300 px 分伸ばす
4. どれでも特定できなければページ全体を渡す（`page`）

切り出しは bbox の上下に 80 px の余白を付け、横幅は全幅にする（単語だけの bbox だと行が途中で切れ、LLM が行の一部だけを見て誤った修正を出すため）。画像は 200 DPI で描画する。
LLM に渡す OCR の抜粋は、該当行と前後の非空行（見つからなければページ全体）。cross_ocr_diff の文脈が複数の行に一致するときは、先頭の行ではなく bbox を含む段落に対応付けた行に絞る（決まらなければページ全体）。

#### 判定と自動適用の基準

LLM の出力は `transcription`（OCR を見ずに画像から書き写した文字列）、`verdict`（`misread` / `correct` / `uncertain`）、`confidence`（`high` / `medium` / `low`）、
`before`（抜粋にそのまま現れる置換前の文字列）、`after`、`reason`。`before` がページ内で複数に一致する場合は、抜粋の位置を手がかりに前後の文字を足して一意にする。
振り分けの規則（`judge/policy.py`）:

| 判定 | 扱い |
|---|---|
| 判定失敗・`uncertain` | 確認待ち |
| `correct`（確信度が low 以外） | 棄却（修正しない） |
| `correct`（確信度 low） | 確認待ち |
| `misread` で、置換前が一意に決まらない／置換が空・変化なし | 確認待ち |
| `misread` で、確信度が `high` 未満・検出器が `katex_error`・変更が 8 文字を超える・一致した判定が `--min-agree` 未満 | 確認待ち（置換の提案は `corrections_pending.json`） |
| 上記以外の `misread` | **自動適用**（`corrections.json`） |

同じ誤読を複数の検出器が挙げた場合は、置換後の文字列が既にあれば重複として 1 件だけ書き出す。`katex_error` を自動適用から外すのは、
描画できない数式（`\cline` など KaTeX が未対応の記法）は誤読とは限らず、LLM が数式を書き換えても原本の誤読を直したことにならないため（上位モデルは実際に書き換えを提案した）。

#### 測定結果（初歩からの数学、人手修正前の Gemini キャッシュ）

- 候補は 314 ページで 135 件（`equation_check` 3・`rare_word` 52・`cross_ocr_diff` 77・`katex_error` 3）。評価セット（33 ページ）には 22 件
- 判定モデルは `gemini-3.5-flash-lite`・`gemini-3.8-flash`・`gemini-3.1-pro-preview` を比較した（単価は `judge/pricing.py`）。詳細は #24 の完了報告コメントを参照
- 上位モデル（`gemini-3.1-pro-preview`）の切り出し画像判定を既定の推奨構成とした。評価セットでは真の誤読 7 件すべてを誤読と判定して正しい修正を出し、自動適用した修正に誤りは無かった
- 314 ページ全体では自動適用 13 件（12 ページ）・確認待ち・棄却に振り分けた。API 料金は 135 候補で約 2.2 USD（1 ページあたり約 0.007 USD）

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
