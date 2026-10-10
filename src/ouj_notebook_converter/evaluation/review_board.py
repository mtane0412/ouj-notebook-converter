"""仕様: 原本画像と正解 Markdown を並べた確認用ページ（校正台）を生成する。

評価セット（manifest.json と正解 page_NNNN.md）の各ページについて、PDF から作った原本画像と
KaTeX で描画した Markdown を 1 枚の HTML にまとめ、人手での確認（OK／要修正＋メモ）に使う。

使い方:
  uv run python -m ouj_notebook_converter.evaluation.review_board \\
      --truth <評価セット> --pdf <原本.pdf> --out <出力先> [--pred <比較対象>]

出力（<出力先>）:
  index.html  KaTeX の JS・CSS（フォント込み）とデータを埋め込んだ 1 ファイル。ローカルで開ける
  data.json   index.html に埋め込んだものと同じデータ（Claude や他のツールが読むための複製）
  img/        page_NNNN.jpg（200 DPI のグレースケール JPEG）。index.html から相対パスで参照する

注意事項:
  - 数式の検出規則は変換時の後処理と同じ MATH_SPAN の正規表現をそのまま data に載せ、ブラウザ側で使う
  - KaTeX の dist は scripts/katex_check/node_modules/katex/dist から取る（無ければ npm ci が必要）。
    Artifact の CSP は外部スタイルシートを読めないため、CSS はフォントごと、JS も含めて埋め込む
  - ページ画像の生成には Pillow が必要（gemini extra に含まれる）。無ければ Fail-Fast する
  - 教材の画像・本文を含む生成物なので、リポジトリには置かない
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer

from ouj_notebook_converter.evaluation.dataset import (
    load_manifest,
    read_prediction,
    read_truth,
)
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN

# scripts/katex_check/package.json の katex と同じバージョン（テストで同期を確認する）
KATEX_VERSION = "0.18.9"
DEFAULT_DPI = 200
JPEG_QUALITY = 80

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KATEX_DIST = _REPO_ROOT / "scripts" / "katex_check" / "node_modules" / "katex" / "dist"
_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "review_board.html"

_PLACEHOLDER_CSS = "/*__KATEX_CSS__*/"
_PLACEHOLDER_JS = "/*__KATEX_JS__*/"
_PLACEHOLDER_DATA = "__DATA_JSON__"

_FONT_URL = re.compile(r"url\((fonts/[^)]+?\.woff2)\)\s*format\(['\"]woff2['\"]\)")
# woff2 以外の src（woff / ttf）。woff2 の埋め込み後は不要なので取り除く
_FALLBACK_FONT_SRC = re.compile(
    r",\s*url\(fonts/[^)]+?\.(?:woff|ttf)\)\s*format\(['\"][^'\"]+['\"]\)"
)

app = typer.Typer(add_completion=False)


class KatexAssetsError(RuntimeError):
    """KaTeX の dist（CSS・フォント・JS）を読み込めなかった場合の例外。"""


def load_katex_assets(dist_dir: Path) -> tuple[str, str]:
    """KaTeX の CSS（woff2 フォントを data URI で埋め込み済み）と JS を読み込む。

    Raises:
        KatexAssetsError: dist や必要なファイルが無い場合（npm ci の実行を案内する）。
    """
    css_path = dist_dir / "katex.min.css"
    js_path = dist_dir / "katex.min.js"
    for path in (css_path, js_path):
        if not path.is_file():
            raise KatexAssetsError(
                f"KaTeX のファイルが見つかりません: {path}\n"
                "scripts/katex_check で `npm ci` を実行してください"
                "（または --katex-dist で katex/dist を指定してください）"
            )

    def embed(match: re.Match[str]) -> str:
        font_path = dist_dir / match.group(1)
        if not font_path.is_file():
            raise KatexAssetsError(f"KaTeX のフォントが見つかりません: {font_path}")
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return f"url(data:font/woff2;base64,{encoded}) format('woff2')"

    css = css_path.read_text(encoding="utf-8")
    css = _FONT_URL.sub(embed, css)
    css = _FALLBACK_FONT_SRC.sub("", css)
    return css, js_path.read_text(encoding="utf-8")


def render_page_images(
    pdf_path: Path, pages: list[int], out_dir: Path, *, dpi: int = DEFAULT_DPI
) -> dict[int, Path]:
    """PDF の指定ページ（1 始まり）をグレースケール JPEG にして書き出す。

    Returns:
        ページ番号から書き出した画像ファイルへの対応。

    Raises:
        FileNotFoundError: PDF が存在しない場合。
        ValueError: PDF に存在しないページが指定された場合。
        RuntimeError: Pillow が未インストールの場合。
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF が見つかりません: {pdf_path}")
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise RuntimeError("ページ画像の生成には Pillow が必要です: uv sync --extra gemini") from e
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    results: dict[int, Path] = {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_number in pages:
            if not 1 <= page_number <= len(pdf):
                raise ValueError(
                    f"PDF にページ {page_number} がありません（全 {len(pdf)} ページ）: {pdf_path}"
                )
            image = pdf[page_number - 1].render(scale=scale).to_pil().convert("L")
            target = out_dir / f"page_{page_number:04d}.jpg"
            image.save(target, format="JPEG", quality=JPEG_QUALITY)
            results[page_number] = target
    finally:
        pdf.close()
    return results


def build_board_data(
    truth_dir: Path,
    pred_dir: Path | None,
    *,
    title: str,
    pred_label: str = "比較対象",
) -> dict[str, Any]:
    """校正台に埋め込むデータ（data.json の内容）を組み立てる。

    ページは manifest.json の記載順に並べる。比較対象を指定した場合は全ページ分が必要。

    Raises:
        FileNotFoundError: 正解・比較対象のページが無い場合。
        ValueError: manifest.json の形式が不正な場合など。
    """
    pages: list[dict[str, Any]] = []
    for entry in load_manifest(truth_dir):
        note_path = truth_dir / "review" / f"page_{entry.page:04d}.md"
        pages.append(
            {
                "page": entry.page,
                "category": entry.category,
                "image": f"img/page_{entry.page:04d}.jpg",
                "truth": read_truth(truth_dir, entry.page),
                "pred": read_prediction(pred_dir, entry.page) if pred_dir is not None else None,
                "notes": (note_path.read_text(encoding="utf-8") if note_path.is_file() else None),
            }
        )
    return {
        "title": title,
        "katexVersion": KATEX_VERSION,
        "mathSpanPattern": MATH_SPAN.pattern,
        "hasPred": pred_dir is not None,
        "predLabel": pred_label,
        "pages": pages,
    }


def render_index_html(template: str, data: dict[str, Any], css: str, js: str) -> str:
    """テンプレートに KaTeX の CSS・JS とデータを埋め込んで index.html の文字列を返す。

    Raises:
        ValueError: テンプレートにプレースホルダが無い場合。
    """
    for placeholder in (_PLACEHOLDER_CSS, _PLACEHOLDER_JS, _PLACEHOLDER_DATA):
        if placeholder not in template:
            raise ValueError(f"テンプレートにプレースホルダ {placeholder} がありません")
    # データ中の "</" と U+2028/2029 を無害化し、<script> 内に安全に埋め込む
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )
    # str.replace は置換文字列中のバックスラッシュを解釈しないので、そのまま差し込める
    return (
        template.replace(_PLACEHOLDER_CSS, css)
        .replace(_PLACEHOLDER_JS, js.replace("</script", "<\\/script"))
        .replace(_PLACEHOLDER_DATA, payload)
    )


def build_review_board(
    truth_dir: Path,
    pdf_path: Path,
    out_dir: Path,
    *,
    pred_dir: Path | None = None,
    katex_dist: Path = DEFAULT_KATEX_DIST,
    title: str = "校正台",
    pred_label: str = "比較対象",
    dpi: int = DEFAULT_DPI,
) -> None:
    """index.html・data.json・img/ を out_dir に書き出す。

    Raises:
        KatexAssetsError: KaTeX の dist を読み込めない場合。
        FileNotFoundError / ValueError / RuntimeError: 入力が不正な場合。
    """
    css, js = load_katex_assets(katex_dist)
    data = build_board_data(truth_dir, pred_dir, title=title, pred_label=pred_label)
    html = render_index_html(_TEMPLATE_PATH.read_text(encoding="utf-8"), data, css, js)
    render_page_images(pdf_path, [p["page"] for p in data["pages"]], out_dir / "img", dpi=dpi)
    (out_dir / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


@app.command()
def main(
    truth: Annotated[
        Path, typer.Option("--truth", help="評価セット（manifest.json と正解 Markdown）")
    ],
    pdf: Annotated[Path, typer.Option("--pdf", help="原本 PDF")],
    out: Annotated[Path, typer.Option("--out", help="出力先ディレクトリ")],
    pred: Annotated[
        Path | None,
        typer.Option(
            "--pred", help="比較対象（下書き・OCR 出力・ページキャッシュ）。差分タブに使う"
        ),
    ] = None,
    pred_label: Annotated[str, typer.Option("--pred-label", help="比較対象の表示名")] = "比較対象",
    title: Annotated[str, typer.Option("--title", help="ページの表題")] = "校正台",
    katex_dist: Annotated[
        Path, typer.Option("--katex-dist", help="katex/dist のディレクトリ")
    ] = DEFAULT_KATEX_DIST,
    dpi: Annotated[int, typer.Option("--dpi", min=1, help="ページ画像の解像度")] = DEFAULT_DPI,
) -> None:
    """原本画像と正解 Markdown（KaTeX 描画）を並べた確認用ページを生成する。"""
    try:
        build_review_board(
            truth,
            pdf,
            out,
            pred_dir=pred,
            katex_dist=katex_dist,
            title=title,
            pred_label=pred_label,
            dpi=dpi,
        )
    except (KatexAssetsError, FileNotFoundError, ValueError, RuntimeError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"校正台を生成しました: {out / 'index.html'}")


if __name__ == "__main__":
    app()
