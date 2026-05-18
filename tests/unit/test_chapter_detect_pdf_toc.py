"""仕様: detect_via_pdf_toc の単体テスト。

pypdfium2 の PdfDocument.get_toc() で PDF のしおりを読んで章境界を検出することを検証する。
reportlab でテスト用 PDF を合成し、pypdfium2 で読み直すシンプルなアプローチ。
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_via_pdf_toc,
)
from ouj_notebook_converter.pipeline.types import ChapterKind, PageMarkdown


def _make_page(page_index: int, markdown: str = "") -> PageMarkdown:
    return PageMarkdown(page_index=page_index, markdown=markdown)


class FakeTocItem(NamedTuple):
    """pypdfium2 の TOC アイテムを模したフェイク。"""
    title: str | None
    page_index: int | None  # 1-origin
    level: int = 0
    is_closed: bool = False
    n_kids: int = 0
    dest: None = None


class FakePdfDocument:
    """pypdfium2.PdfDocument の最小フェイク。"""

    def __init__(self, toc_items: list[FakeTocItem]) -> None:
        self._toc_items = toc_items

    def get_toc(self) -> list[FakeTocItem]:
        return self._toc_items

    def __len__(self) -> int:
        return 20


class TestDetectViaPdfToc:
    def test_しおりから章タイトルとページ境界が取得できる(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toc = [
            FakeTocItem(title="まえがき", page_index=1),
            FakeTocItem(title="第1章 データとは何か", page_index=3),
            FakeTocItem(title="第2章 データの収集と前処理", page_index=8),
            FakeTocItem(title="あとがき", page_index=18),
        ]
        _patch_pdfium(monkeypatch, toc)

        pages = [_make_page(i) for i in range(20)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        chapters = detect_via_pdf_toc(dummy_pdf, pages)

        kinds = [c.kind for c in chapters]
        assert ChapterKind.PREFACE in kinds
        assert ChapterKind.CHAPTER in kinds
        assert ChapterKind.AFTERWORD in kinds

    def test_章タイトルが正しく取得される(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toc = [
            FakeTocItem(title="第1章 データとは何か", page_index=1),
        ]
        _patch_pdfium(monkeypatch, toc)

        pages = [_make_page(i) for i in range(5)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        chapters = detect_via_pdf_toc(dummy_pdf, pages)
        chapter_items = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_items) == 1
        assert chapter_items[0].title == "データとは何か"
        assert chapter_items[0].chapter_number == 1

    def test_ページ境界が正しく計算される(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toc = [
            FakeTocItem(title="第1章 はじめに", page_index=1),
            FakeTocItem(title="第2章 応用", page_index=6),
        ]
        _patch_pdfium(monkeypatch, toc)

        pages = [_make_page(i) for i in range(10)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        chapters = detect_via_pdf_toc(dummy_pdf, pages)
        chapter_items = sorted(
            [c for c in chapters if c.kind == ChapterKind.CHAPTER],
            key=lambda c: c.chapter_number or 0,
        )
        assert chapter_items[0].start_page_index == 0   # page_index=1 → 0-origin
        assert chapter_items[0].end_page_index == 4
        assert chapter_items[1].start_page_index == 5
        assert chapter_items[1].end_page_index == 9

    def test_しおりが空の場合はエラー(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_pdfium(monkeypatch, [])

        pages = [_make_page(i) for i in range(5)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        with pytest.raises(ChapterDetectionError, match="しおり"):
            detect_via_pdf_toc(dummy_pdf, pages)

    def test_しおりに章認識可能エントリがない場合はエラー(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toc = [
            FakeTocItem(title="表紙", page_index=1),
            FakeTocItem(title="奥付", page_index=20),
        ]
        _patch_pdfium(monkeypatch, toc)

        pages = [_make_page(i) for i in range(20)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        with pytest.raises(ChapterDetectionError):
            detect_via_pdf_toc(dummy_pdf, pages)

    def test_sourceはpdf_toc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toc = [FakeTocItem(title="第1章 データとは何か", page_index=1)]
        _patch_pdfium(monkeypatch, toc)

        pages = [_make_page(i) for i in range(5)]
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        chapters = detect_via_pdf_toc(dummy_pdf, pages)
        assert all(c.source == "pdf_toc" for c in chapters)


def _patch_pdfium(monkeypatch: pytest.MonkeyPatch, toc: list[FakeTocItem]) -> None:
    """pypdfium2.PdfDocument を FakePdfDocument に差し替える。"""
    import ouj_notebook_converter.pipeline.stages.chapter_detect as mod

    class FakeModule:
        class PdfDocument(FakePdfDocument):
            def __init__(self, path: str) -> None:
                super().__init__(toc)

    monkeypatch.setattr(mod, "_get_pdfium", lambda: FakeModule)
