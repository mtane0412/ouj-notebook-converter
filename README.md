# ouj-notebook-converter

放送大学の PDF テキストを Markdown（および epub/PDF/txt）に変換する CLI ツール。

## 機能

- yomitoku による OCR（日本語対応レイアウト解析）
- 数式変換（pix2tex / Pix2Text の 2 バックエンドから選択可能）
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
# PDF を Markdown に変換（カレントディレクトリに出力）
uv run ounc 放送大学テキスト.pdf

# 出力先を指定
uv run ounc 放送大学テキスト.pdf --outdir /tmp/output

# ページ範囲を指定（1 ページ目、3〜5 ページ目、10 ページ目）
uv run ounc 放送大学テキスト.pdf --pages 1,3-5,10

# 章ごとにファイルを分割
uv run ounc 放送大学テキスト.pdf --split chapters
```

## 数式変換

数式を含む PDF には `--math-backend` オプションを使用する。

| バックエンド | 概要 | 推奨シナリオ |
|---|---|---|
| `none` | 数式変換なし（デフォルト） | 数式のない PDF |
| `pix2tex` | yomitoku の paragraph ロール経由で pix2tex API を呼び出す | — |
| `pix2text` | Pix2Text でページ全体から数式を検出・認識する | 数式を含む PDF（推奨） |

> **注意**: yomitoku 0.13.0 の layout parser モデル（`rtdetrv2v2`）は formula カテゴリを持たないため、`pix2tex` バックエンドでは数式が検出されない。**数式変換には `pix2text` バックエンドを推奨する。**

### Pix2Text バックエンドのセットアップ

Pix2Text は yomitoku と依存関係が競合する可能性があるため、専用の venv で動かす。

#### 1. Pix2Text venv の作成

```bash
python3.11 -m venv ~/.venvs/pix2text
~/.venvs/pix2text/bin/pip install "pix2text[serve]"
```

#### 2. 自前ラッパーサーバーの起動（別ターミナルで）

```bash
~/.venvs/pix2text/bin/python scripts/pix2text_server.py --port 8503
```

起動確認:

```bash
curl http://localhost:8503/health
# → {"ok":true}
```

#### 3. 変換の実行

```bash
uv run ounc 数式入りPDF.pdf --outdir /tmp/output --math-backend pix2text
```

カスタム URL を使う場合:

```bash
uv run ounc 数式入りPDF.pdf --outdir /tmp/output \
    --math-backend pix2text \
    --pix2text-url http://localhost:9000
```

### pix2tex バックエンド（参考）

yomitoku で formula ロールが付いた paragraph を pix2tex で LaTeX 化する経路（現行モデルでは検出されないため実用外）。

```bash
# pix2tex サーバーを起動した上で
uv run ounc 数式入りPDF.pdf --math-backend pix2tex --pix2tex-url http://localhost:8502
```

## オプション一覧

```
Usage: ounc [OPTIONS] INPUT_PDF

Arguments:
  input_pdf  変換する PDF ファイルのパス

Options:
  -o, --outdir PATH                出力先ディレクトリ
  -f, --format [md|epub|pdf|txt]   出力形式（複数指定可）[default: md]
  -d, --device TEXT                推論デバイス: mps / cpu / cuda [default: mps]
      --dpi INTEGER                PDF レンダリング DPI [default: 200]
      --pages TEXT                 処理するページ範囲 例: 1,3-5,10
      --cache-dir PATH             キャッシュディレクトリ
      --no-cache                   キャッシュを無効化
      --combine/--no-combine       全ページを 1 ファイルに結合 [default: combine]
      --reading-order [auto|...]   読み順推定モード [default: auto]
      --ignore-meta/--no-ignore-meta  ヘッダ/フッタを除外 [default: ignore-meta]
      --split [none|chapters]      出力分割モード [default: none]
      --math-backend [none|pix2tex|pix2text]  数式変換バックエンド [default: none]
      --pix2tex-url TEXT           pix2tex API サーバー URL [default: http://localhost:8502]
      --pix2text-url TEXT          Pix2Text ラッパー URL [default: http://localhost:8503]
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
    │  math_backend=pix2tex  ──────────────────────▶ math_extract
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

- `math_extract`: yomitoku paragraph の role（inline_formula / display_formula）を見て pix2tex で LaTeX 化
- `math_detect`: Pix2Text でページ全体を解析し、yomitoku paragraph と IoU マッチして LaTeX 化
