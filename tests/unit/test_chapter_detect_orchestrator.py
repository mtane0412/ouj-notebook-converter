"""仕様: detect_chapters オーケストレーターの単体テスト。

フォールバックチェーン（syllabus → pdf_toc → ocr_toc → body_headings）が
正しい順序で試行されることを検証する。各検出器はモックで差し替える。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_chapters,
)
from ouj_notebook_converter.pipeline.types import ChapterKind, ChapterSpec, PageMarkdown

_MODULE = "ouj_notebook_converter.pipeline.stages.chapter_detect"


def _make_page(page_index: int) -> PageMarkdown:
    return PageMarkdown(page_index=page_index, markdown="")


def _make_chapter_spec() -> list[ChapterSpec]:
    return [
        ChapterSpec(
            order=0,
            kind=ChapterKind.CHAPTER,
            chapter_number=1,
            title="テスト章",
            start_page_index=0,
            end_page_index=4,
            source="test",
        )
    ]


class Testフォールバックチェーン:
    def test_course_code指定時はシラバス検出が最初に試行される(
        self, tmp_path: Path
    ) -> None:
        expected = _make_chapter_spec()
        with patch(f"{_MODULE}.detect_via_syllabus", return_value=expected) as mock_syllabus:
            pages = [_make_page(i) for i in range(5)]
            result = detect_chapters(
                tmp_path / "dummy.pdf", pages, course_code="1234567"
            )
        mock_syllabus.assert_called_once()
        assert result == expected

    def test_シラバス失敗時はpdf_tocへフォールバック(self, tmp_path: Path) -> None:
        expected = _make_chapter_spec()
        with (
            patch(f"{_MODULE}.detect_via_syllabus", side_effect=ChapterDetectionError("失敗")),
            patch(f"{_MODULE}.detect_via_pdf_toc", return_value=expected) as mock_pdf,
        ):
            pages = [_make_page(i) for i in range(5)]
            result = detect_chapters(
                tmp_path / "dummy.pdf", pages, course_code="1234567"
            )
        mock_pdf.assert_called_once()
        assert result == expected

    def test_course_code未指定時はシラバス検出をスキップする(
        self, tmp_path: Path
    ) -> None:
        expected = _make_chapter_spec()
        with (
            patch(f"{_MODULE}.detect_via_syllabus") as mock_syllabus,
            patch(f"{_MODULE}.detect_via_pdf_toc", return_value=expected),
        ):
            pages = [_make_page(i) for i in range(5)]
            detect_chapters(tmp_path / "dummy.pdf", pages, course_code=None)
        mock_syllabus.assert_not_called()

    def test_pdf_toc失敗時はocr_tocへフォールバック(self, tmp_path: Path) -> None:
        expected = _make_chapter_spec()
        with (
            patch(f"{_MODULE}.detect_via_pdf_toc", side_effect=ChapterDetectionError("失敗")),
            patch(f"{_MODULE}.detect_via_ocr_toc", return_value=expected) as mock_ocr,
        ):
            pages = [_make_page(i) for i in range(5)]
            result = detect_chapters(tmp_path / "dummy.pdf", pages)
        mock_ocr.assert_called_once()
        assert result == expected

    def test_ocr_toc失敗時はbody_headingsへフォールバック(self, tmp_path: Path) -> None:
        expected = _make_chapter_spec()
        with (
            patch(f"{_MODULE}.detect_via_pdf_toc", side_effect=ChapterDetectionError("失敗")),
            patch(f"{_MODULE}.detect_via_ocr_toc", side_effect=ChapterDetectionError("失敗")),
            patch(f"{_MODULE}.detect_via_body_headings", return_value=expected) as mock_body,
        ):
            pages = [_make_page(i) for i in range(5)]
            result = detect_chapters(tmp_path / "dummy.pdf", pages)
        mock_body.assert_called_once()
        assert result == expected

    def test_全検出器失敗時はChapterDetectionErrorを送出する(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(f"{_MODULE}.detect_via_pdf_toc", side_effect=ChapterDetectionError("TOC失敗")),
            patch(f"{_MODULE}.detect_via_ocr_toc", side_effect=ChapterDetectionError("OCR失敗")),
            patch(
                f"{_MODULE}.detect_via_body_headings",
                side_effect=ChapterDetectionError("本文失敗"),
            ),
        ):
            pages = [_make_page(i) for i in range(5)]
            with pytest.raises(ChapterDetectionError, match="章境界を検出できません"):
                detect_chapters(tmp_path / "dummy.pdf", pages)

    def test_エラーメッセージに各検出器の失敗理由が含まれる(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(f"{_MODULE}.detect_via_pdf_toc", side_effect=ChapterDetectionError("TOC失敗")),
            patch(f"{_MODULE}.detect_via_ocr_toc", side_effect=ChapterDetectionError("OCR失敗")),
            patch(
                f"{_MODULE}.detect_via_body_headings",
                side_effect=ChapterDetectionError("本文失敗"),
            ),
        ):
            pages = [_make_page(i) for i in range(5)]
            with pytest.raises(ChapterDetectionError) as exc_info:
                detect_chapters(tmp_path / "dummy.pdf", pages)

        error_msg = str(exc_info.value)
        assert "TOC失敗" in error_msg
        assert "OCR失敗" in error_msg
        assert "本文失敗" in error_msg
