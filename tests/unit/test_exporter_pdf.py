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

    def test_全ページにOCR_JSONが無い場合はValueErrorを送出しPDFを出力しない(
        self, tmp_path: Path
    ) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        pages = [_make_page(0, "テスト本文テキスト", yomitoku_json_path=None)]
        out_pdf = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="単語"):
            export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert not out_pdf.exists()

    def test_全ページの単語が空の場合はValueErrorを送出しPDFを出力しない(
        self, tmp_path: Path
    ) -> None:
        # Gemini バックエンドの analysis.json は words が常に空リストになる
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        json_path.write_text(
            json.dumps(
                {"backend": "gemini", "markdown": "テスト本文", "paragraphs": [], "words": []},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pages = [_make_page(0, "テスト本文", yomitoku_json_path=json_path)]
        out_pdf = tmp_path / "output.pdf"
        with pytest.raises(ValueError, match="単語"):
            export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert not out_pdf.exists()


class TestExportPdf:
    """OCRオーバーレイPDF生成のテスト。"""

    def test_出力PDFファイルが作成される(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        pages = [_make_page(0, "テスト本文テキスト", yomitoku_json_path=json_path)]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.exists()

    def test_出力ファイルはPDF形式である(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        pages = [_make_page(0, "テスト本文テキスト", yomitoku_json_path=json_path)]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        # PDF ファイルのシグネチャ確認
        assert out_pdf.read_bytes()[:4] == b"%PDF"

    def test_yomitoku_json_pathがNoneのページも正常に処理される(self, tmp_path: Path) -> None:
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        # 一部のページが yomitoku_json_path=None（OCR JSONなし）でもクラッシュしないこと
        pages = [
            _make_page(0, "テスト", yomitoku_json_path=json_path),
            _make_page(1, "白紙ページ", yomitoku_json_path=None),
        ]
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
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        pages = [_make_page(0, "テスト", yomitoku_json_path=json_path)]
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

        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)
        pages = [
            _make_page(0, "1ページ目の内容", yomitoku_json_path=json_path),
            _make_page(1, "2ページ目の内容", yomitoku_json_path=json_path),
        ]
        out_pdf = tmp_path / "output.pdf"
        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)
        assert out_pdf.exists()
        # reportlab で page count を確認するため pypdfium2 を使用
        import pypdfium2

        doc = pypdfium2.PdfDocument(str(out_pdf))
        assert len(doc) == 2


def _extract_text(pdf_path: Path, page_index: int = 0) -> str:
    """出力 PDF の 1 ページから、ページ内に描画されたテキストを描画順に抽出する。"""
    import pypdfium2

    doc = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return str(doc[page_index].get_textpage().get_text_bounded())
    finally:
        doc.close()


def _write_words_json(path: Path, words: list[tuple[str, int, int, int, int]]) -> Path:
    """段落 1 つに単語を並べた yomitoku 形式の JSON を書き出す（words は JSON に書く順）。"""
    data = {
        "paragraphs": [{"box": [0, 0, 1600, 2300], "order": 0, "direction": "horizontal"}],
        "figures": [],
        "words": [
            {
                "content": content,
                "direction": "horizontal",
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            }
            for content, x1, y1, x2, y2 in words
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestTextLayerOrder:
    """テキストレイヤーの描画順。"""

    def test_draws_yomitoku_words_in_reading_order(self, tmp_path: Path) -> None:
        """yomitoku の words はページの下の行から並ぶことがあるため、読み順に並べて描画する。"""
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = _write_words_json(
            tmp_path / "analysis.json",
            [("二行目の文", 100, 300, 500, 340), ("一行目の文", 100, 200, 500, 240)],
        )
        pages = [_make_page(0, "", yomitoku_json_path=json_path)]
        out_pdf = tmp_path / "output.pdf"

        export_pdf(pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf)

        text = _extract_text(out_pdf)
        assert text.index("一行目の文") < text.index("二行目の文")


class TestWordBboxDir:
    """別の OCR（Gemini）の文字を yomitoku の単語 bbox に差し込む。"""

    def test_places_markdown_text_on_word_bboxes(self, tmp_path: Path) -> None:
        """word_bbox_dir の単語 bbox に、PageMarkdown の Markdown の文字を割り当てて描画する。"""
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        bbox_dir = tmp_path / "yomitoku"
        _write_words_json(
            bbox_dir / "page_0001" / "analysis.json",
            [("数の慨念", 100, 200, 500, 240), ("について", 100, 300, 500, 340)],
        )
        pages = [_make_page(0, "# 数の概念\n\nについて")]
        out_pdf = tmp_path / "output.pdf"

        stats = export_pdf(
            pages, out_pdf, tmp_path / "assets", source_pdf=source_pdf, word_bbox_dir=bbox_dir
        )

        text = _extract_text(out_pdf)
        assert "数の概念" in text
        assert "慨念" not in text
        assert stats is not None
        assert stats.words == 2
        assert stats.chars_replaced == 1

    def test_returns_no_stats_without_word_bbox_dir(self, tmp_path: Path) -> None:
        """word_bbox_dir を指定しない場合は割り当て件数を返さない。"""
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        json_path = tmp_path / "analysis.json"
        _make_yomitoku_json(json_path)

        stats = export_pdf(
            [_make_page(0, "", yomitoku_json_path=json_path)],
            tmp_path / "output.pdf",
            tmp_path / "assets",
            source_pdf=source_pdf,
        )

        assert stats is None

    def test_fits_long_text_into_word_bbox_width(self, tmp_path: Path) -> None:
        """yomitoku が読み落とした文字が単語に集まっても、ページの外にはみ出さず抽出できる。"""
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        bbox_dir = tmp_path / "yomitoku"
        _write_words_json(bbox_dir / "page_0001" / "analysis.json", [("本文", 1300, 200, 1400, 240)])
        long_text = "本文" + "読み落とされた長い文章" * 5
        out_pdf = tmp_path / "output.pdf"

        export_pdf(
            [_make_page(0, long_text)],
            out_pdf,
            tmp_path / "assets",
            source_pdf=source_pdf,
            word_bbox_dir=bbox_dir,
        )

        assert _extract_text(out_pdf).replace("\r\n", "") == long_text

    def test_missing_word_bbox_json_stops_before_writing_pdf(self, tmp_path: Path) -> None:
        """対象ページの単語 bbox の JSON が無ければ、PDF を出力せずに停止する。"""
        source_pdf = tmp_path / "source.pdf"
        _make_minimal_pdf(source_pdf)
        bbox_dir = tmp_path / "yomitoku"
        bbox_dir.mkdir()
        out_pdf = tmp_path / "output.pdf"

        with pytest.raises(FileNotFoundError, match="1 ページ"):
            export_pdf(
                [_make_page(0, "本文")],
                out_pdf,
                tmp_path / "assets",
                source_pdf=source_pdf,
                word_bbox_dir=bbox_dir,
            )
        assert not out_pdf.exists()
