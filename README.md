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
