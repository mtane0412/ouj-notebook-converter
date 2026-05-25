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

# モデルを変更する場合（デフォルト: gemini-3.5-flash）
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output --ocr-backend gemini --gemini-model gemini-2.0-flash
```

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
      --gemini-model TEXT                     Gemini モデル名 [default: gemini-3.5-flash]
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
