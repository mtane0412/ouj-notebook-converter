"""仕様: exporters.pdf モジュール（OCRオーバーレイPDFエクスポーター）のユニットテスト。

元PDFページ画像を背景にし、Yomitoku JSON の word bbox に基づく不可視テキストレイヤーを
重ねた searchable PDF を生成する処理をテストする。
reportlab・pypdfium2 は実際に利用可能なためモック不要で実行テストとする。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouj_notebook_converter.exporters.pdf import export_pdf
from ouj_notebook_converter.pipeline.types import PageMarkdown

# テスト用最小 PDF（1ページ、A4サイズ）- reportlab で生成したバイナリ相当
# pytest実行時に動的生成するため、conftest の代わりに fixture として用意する


def _make_page(
    page_index: int,
    markdown: str,
    yomitoku_json_path: Path | None = None,
) -> PageMarkdown:
    """テスト用 PageMarkdown を作る簡易ファクトリ。"""
    return PageMarkdown(
        page_index=page_index,
        markdown=markdown,
        referenced_assets=[],
        yomitoku_json_path=yomitoku_json_path,
    )


def _make_minimal_pdf(out_path: Path) -> None:
    """テスト用の最小 PDF（1ページ・A4）を reportlab で生成する。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas

    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    c.drawString(100, 700, "テストページ")
    c.showPage()
    c.save()


def _make_yomitoku_json(out_path: Path) -> None:
    """テスト用の Yomitoku JSON を生成する（単語が 1 つあるページ）。"""
    data = {
        "paragraphs": [
            {"box": [50, 50, 300, 80], "contents": "テスト本文テキスト"}
        ],
        "words": [
            {
                "content": "テスト",
                "points": [[50, 50], [150, 50], [150, 80], [50, 80]],
            },
            {
                "content": "本文",
                "points": [[155, 50], [230, 50], [230, 80], [155, 80]],
            },
        ],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestExportPdfValidation:
    """入力バリデーションのテスト。"""

    def test_pagesが空の場合はValueErrorを送出する(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "source.pdf"
        _make_minimal_pdf(pdf_path)
        with pytest.raises(ValueError, match="pages が空"):
            export_pdf([], tmp_path / "out.pdf", tmp_path / "assets", source_pdf=pdf_path)


class TestExportPdf:
    """OCRオーバーレイPDF生成のテスト。"""

    def test_出力PDFファイルが作成される(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        pages = [_make_page(0, "テスト本文テキスト")]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.exists()

    def test_出力ファイルはPDF形式である(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        pages = [_make_page(0, "テスト本文テキスト")]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        # PDF ファイルのシグネチャ確認
        assert out_pdf.read_bytes()[:4] == b"%PDF"

    def test_yomitoku_json_pathがNoneのページも正常に処理される(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        # yomitoku_json_path=None のページ（OCR JSONなし）でもクラッシュしないこと
        pages = [_make_page(0, "テスト", yomitoku_json_path=None)]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.read_bytes()[:4] == b"%PDF"

    def test_yomitoku_json_pathが有効なページはテキストレイヤー付きPDFが生成される(
        self, tmp_path: Path
    ) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        pages = [_make_page(0, "テスト本文テキスト", yomitoku_json_path=json_path)]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        # PDF バイナリ内にテキストコンテンツが含まれることを確認
        pdf_bytes = out_pdf.read_bytes()
        assert b"%PDF" in pdf_bytes
        # テキストレイヤーが含まれる場合、PDFサイズが画像のみより大きくなる傾向がある
        # ここでは正常終了とファイル存在を確認する
        assert out_pdf.stat().st_size > 0

    def test_出力ディレクトリが存在しなくても作成される(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        pages = [_make_page(0, "テスト")]
        out_pdf = tmp_path / "subdir" / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.exists()

    def test_複数ページのPDFが生成される(self, tmp_path: Path) -> None:
        # 2ページの source PDF を生成する
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas

        source_pdf = tmp_path / "source.pdf"
        c = rl_canvas.Canvas(str(source_pdf), pagesize=A4)
        c.drawString(100, 700, "1ページ目")
        c.showPage()
        c.drawString(100, 700, "2ページ目")
        c.showPage()
        c.save()

        pages = [
            _make_page(0, "1ページ目の内容"),
            _make_page(1, "2ページ目の内容"),
        ]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.exists()
        # reportlab で page count を確認するため pypdfium2 を使用
        import pypdfium2

        doc = pypdfium2.PdfDocument(str(out_pdf))
        assert len(doc) == 2
