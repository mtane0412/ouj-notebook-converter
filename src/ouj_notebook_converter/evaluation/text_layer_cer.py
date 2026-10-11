"""仕様: searchable PDF のテキストレイヤーを評価セットの正解と比べ、文字誤り率（CER）を出す（issue #42）。

評価セットのページだけで searchable PDF を作り、PDF から抽出したテキストを正解と比べる。
比べるテキストレイヤーは次のとおり。
  - yomitoku: yomitoku の単語の文字をそのまま読み順に描画したもの
  - gemini: yomitoku の単語 bbox に、Gemini の Markdown（人手修正・後処理の適用後）の文字を割り当てたもの
  - legacy（--legacy-pdf を指定した場合）: 既存の searchable PDF の該当ページ

使い方:
  uv run python -m ouj_notebook_converter.evaluation.text_layer_cer \\
    --truth <評価セット> --pdf <元 PDF> --word-bbox-cache <yomitoku のページキャッシュ> \\
    --gemini-cache <Gemini のページキャッシュ> [--corrections <修正ファイル>] \\
    [--legacy-pdf <既存の searchable PDF>] --out-dir <PDF の出力先> [--json <結果 JSON>]

指標:
  - full: 正解 Markdown を、テキストレイヤーと同じ規則（記法を除き数式を平文化、NFKC・空白除去）で
    文字列にしたものとの CER。数式の平文化の規則が Gemini 側と同じため、数式の多いページでは
    Gemini 側に有利に出る
  - japanese: かな・漢字・長音符・々だけを残して比べた CER。数式・記号の書き方の違いを数えない

注意事項:
  - 評価セットの正解は Gemini の出力を下書きにしているため、Gemini 側の CER は低く出やすい
  - PDF からの抽出は pypdfium2 の get_text_bounded（ページ内の文字を描画順に返す）で行う
  - 修正ファイルは書籍全体を対象にするため、評価セットに無いページの修正は適用しない。
    そのページ番号は無視せず、標準出力と結果 JSON（corrections_outside_eval）に報告する
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from ouj_notebook_converter.corrections import load_corrections
from ouj_notebook_converter.evaluation.dataset import load_manifest, read_truth
from ouj_notebook_converter.evaluation.metrics import CerScore, levenshtein
from ouj_notebook_converter.exporters.pdf import export_pdf, word_bbox_json_path
from ouj_notebook_converter.exporters.text_layer import AlignmentStats, markdown_to_search_text
from ouj_notebook_converter.pipeline.stages.post_process import build_page_markdown
from ouj_notebook_converter.pipeline.types import PageAnalysis, PageMarkdown

app = typer.Typer(add_completion=False)

# 日本語の文字: ひらがな・カタカナ・漢字・々・〆・長音符
_JAPANESE_CHARS = re.compile(r"[ぁ-ゟァ-ヿ㐀-鿿々〆ー]")
_WHITESPACE = re.compile(r"\s+")
_VARIANTS = ("yomitoku", "gemini")


@dataclass(frozen=True)
class TextLayerScore:
    """1 ページ（または合計）のテキストレイヤーの CER の元になる件数。

    Attributes:
        full: 記法を除き数式を平文化した全文の比較。
        japanese: かな・漢字だけの比較。
    """

    full: CerScore
    japanese: CerScore

    def __add__(self, other: TextLayerScore) -> TextLayerScore:
        return TextLayerScore(
            full=_add_cer(self.full, other.full),
            japanese=_add_cer(self.japanese, other.japanese),
        )


def japanese_only(text: str) -> str:
    """かな・漢字・長音符・々・〆だけを残す。"""
    return "".join(_JAPANESE_CHARS.findall(text))


def score_text_layer(truth_markdown: str, layer_text: str) -> TextLayerScore:
    """正解 Markdown と、PDF から抽出したテキストレイヤーの文字列を比べる。

    Args:
        truth_markdown: 1 ページ分の正解 Markdown。
        layer_text: PDF から抽出したテキスト（改行・空白を含んでよい）。

    Returns:
        全文と日本語だけの CER の元になる件数。
    """
    truth = markdown_to_search_text(truth_markdown)
    pred = _WHITESPACE.sub("", unicodedata.normalize("NFKC", layer_text))
    truth_japanese = japanese_only(truth)
    return TextLayerScore(
        full=CerScore(edits=levenshtein(truth, pred), truth_length=len(truth)),
        japanese=CerScore(
            edits=levenshtein(truth_japanese, japanese_only(pred)),
            truth_length=len(truth_japanese),
        ),
    )


def extract_page_texts(pdf_path: Path, page_indices: list[int]) -> list[str]:
    """PDF の指定ページ（0 始まり）から、ページ内に描画されたテキストを描画順に抽出する。"""
    import pypdfium2

    doc = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return [str(doc[i].get_textpage().get_text_bounded()) for i in page_indices]
    finally:
        doc.close()


@app.command()
def main(
    truth: Annotated[Path, typer.Option(help="評価セットのディレクトリ（manifest.json を含む）")],
    pdf: Annotated[Path, typer.Option(help="元 PDF（背景画像に使う）")],
    word_bbox_cache: Annotated[
        Path, typer.Option(help="yomitoku のページキャッシュ（page_NNNN/analysis.json を含む）")
    ],
    gemini_cache: Annotated[
        Path, typer.Option(help="Gemini のページキャッシュ（page_NNNN/raw.md を含む）")
    ],
    out_dir: Annotated[Path, typer.Option(help="評価用の searchable PDF の出力先")],
    corrections: Annotated[
        Path | None, typer.Option(help="Gemini の出力に適用する人手修正ファイル")
    ] = None,
    legacy_pdf: Annotated[
        Path | None,
        typer.Option(help="あわせて評価する既存の searchable PDF（書籍の全ページを含むもの）"),
    ] = None,
    dpi: Annotated[int, typer.Option(help="OCR 時のレンダリング DPI")] = 200,
    json_path: Annotated[Path | None, typer.Option("--json", help="結果を書き出す JSON")] = None,
) -> None:
    """評価セットのページで、yomitoku と Gemini のテキストレイヤーの CER を比べる。"""
    eval_pages = load_manifest(truth)
    page_corrections = load_corrections(corrections) if corrections else {}
    # 評価ページ以外の修正は適用されないため、ページ番号を報告用に控える
    corrections_outside_eval = sorted(set(page_corrections) - {p.page for p in eval_pages})

    yomitoku_pages = [
        PageMarkdown(
            page_index=p.page - 1,
            markdown="",
            yomitoku_json_path=word_bbox_json_path(word_bbox_cache, p.page - 1),
        )
        for p in eval_pages
    ]
    gemini_pages = [
        build_page_markdown(
            PageAnalysis(
                page_index=p.page - 1,
                yomitoku_json_path=gemini_cache / f"page_{p.page:04d}" / "analysis.json",
                markdown_raw_path=gemini_cache / f"page_{p.page:04d}" / "raw.md",
            ),
            corrections=page_corrections.get(p.page, ()),
        )
        for p in eval_pages
    ]

    assets_dir = out_dir / "assets"
    export_pdf(yomitoku_pages, out_dir / "yomitoku.pdf", assets_dir, source_pdf=pdf, dpi=dpi)
    alignment = export_pdf(
        gemini_pages,
        out_dir / "gemini.pdf",
        assets_dir,
        source_pdf=pdf,
        dpi=dpi,
        word_bbox_dir=word_bbox_cache,
    )
    if alignment is None:
        raise RuntimeError("word_bbox_dir を指定したのに割り当て件数が返りませんでした。")

    # PDF は評価ページだけを順に含む。既存の searchable PDF は書籍の全ページを含む
    sequential = list(range(len(eval_pages)))
    texts = {name: extract_page_texts(out_dir / f"{name}.pdf", sequential) for name in _VARIANTS}
    if legacy_pdf is not None:
        texts["legacy"] = extract_page_texts(legacy_pdf, [p.page - 1 for p in eval_pages])

    rows: list[dict[str, Any]] = []
    totals: dict[str, TextLayerScore] = {}
    for index, eval_page in enumerate(eval_pages):
        truth_markdown = read_truth(truth, eval_page.page)
        row: dict[str, Any] = {"page": eval_page.page, "category": eval_page.category}
        for name, page_texts in texts.items():
            score = score_text_layer(truth_markdown, page_texts[index])
            totals[name] = totals[name] + score if name in totals else score
            row[name] = _score_dict(score)
        rows.append(row)

    report = {
        "pages": rows,
        "total": {name: _score_dict(score) for name, score in totals.items()},
        "alignment": _alignment_dict(alignment),
        "corrections_outside_eval": corrections_outside_eval,
    }
    if json_path is not None:
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, score in totals.items():
        typer.echo(
            f"{name}: 全文 CER {score.full.cer:.2%}（{score.full.edits}/{score.full.truth_length}）"
            f" / 日本語 CER {score.japanese.cer:.2%}"
            f"（{score.japanese.edits}/{score.japanese.truth_length}）"
        )
    typer.echo(f"割り当て件数: {_alignment_dict(alignment)}")
    if corrections_outside_eval:
        typer.echo(
            "評価対象外のため適用しなかった修正のページ: "
            + ", ".join(str(page) for page in corrections_outside_eval)
        )


def _add_cer(a: CerScore, b: CerScore) -> CerScore:
    return CerScore(edits=a.edits + b.edits, truth_length=a.truth_length + b.truth_length)


def _score_dict(score: TextLayerScore) -> dict[str, Any]:
    return {
        part: {"edits": cer.edits, "truth_length": cer.truth_length, "cer": round(cer.cer, 4)}
        for part, cer in (("full", score.full), ("japanese", score.japanese))
    }


def _alignment_dict(stats: AlignmentStats) -> dict[str, int]:
    return asdict(stats)


if __name__ == "__main__":
    app()
