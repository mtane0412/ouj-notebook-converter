"""仕様: detect_via_body_headings の単体テスト。

各ページの Markdown 冒頭から 第N章・まえがき・あとがき・索引 のパターンを
検出できることを検証する。
"""
import pytest

from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_via_body_headings,
)
from ouj_notebook_converter.pipeline.types import ChapterKind, PageMarkdown


def _make_page(page_index: int, markdown: str) -> PageMarkdown:
    return PageMarkdown(page_index=page_index, markdown=markdown)


class Test第N章パターン検出:
    def test_第1章が検出される(self) -> None:
        pages = [
            _make_page(0, "# まえがき\n本書について"),
            _make_page(1, "# 第1章 データとは何か\n本文..."),
            _make_page(2, "続き"),
            _make_page(3, "# 第2章 データの収集\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_kinds = [c.kind for c in chapters]
        assert ChapterKind.CHAPTER in chapter_kinds

    def test_章タイトルが取得される(self) -> None:
        pages = [
            _make_page(0, "# 第1章 データとは何か\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_chapters) == 1
        assert chapter_chapters[0].title == "データとは何か"

    def test_章番号が取得される(self) -> None:
        pages = [
            _make_page(0, "# 第3章 統計的検定\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].chapter_number == 3

    def test_全角数字の章番号が取得される(self) -> None:
        pages = [
            _make_page(0, "# 第１章 データとは何か\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].chapter_number == 1

    def test_漢数字の章番号が取得される(self) -> None:
        pages = [
            _make_page(0, "# 第一章 データとは何か\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].chapter_number == 1

    def test_章見出しがページ中段に現れる場合は検出しない(self) -> None:
        """ページ冒頭以外に現れる章見出しはノイズとして無視する。"""
        pages = [
            _make_page(0, "# はじめに\n第1章について説明する。\n\n## 第1章 データとは何か"),
        ]
        # ページ冒頭の H1 が「はじめに」なのでこれは PREFACE として検出される
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_chapters) == 0

    def test_複数章のページ範囲が正しい(self) -> None:
        pages = [
            _make_page(0, "# まえがき\n本書について"),
            _make_page(1, "# 第1章 データとは何か\n本文..."),
            _make_page(2, "第1章の続き"),
            _make_page(3, "# 第2章 データの収集\n本文..."),
            _make_page(4, "第2章の続き"),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = sorted(
            [c for c in chapters if c.kind == ChapterKind.CHAPTER],
            key=lambda c: c.chapter_number or 0,
        )
        assert chapter_chapters[0].start_page_index == 1
        assert chapter_chapters[0].end_page_index == 2
        assert chapter_chapters[1].start_page_index == 3
        assert chapter_chapters[1].end_page_index == 4

    def test_sourceはbody_headings(self) -> None:
        pages = [_make_page(0, "# 第1章 データとは何か\n本文...")]
        chapters = detect_via_body_headings(pages)
        assert all(c.source == "body_headings" for c in chapters)


class Testまえがきあとがき索引:
    def test_まえがきが検出される(self) -> None:
        pages = [
            _make_page(0, "# まえがき\n本書について"),
            _make_page(1, "# 第1章 データとは何か\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert len(prefaces) == 1

    def test_はじめにがPREFACEとして検出される(self) -> None:
        pages = [
            _make_page(0, "# はじめに\n本書について"),
            _make_page(1, "# 第1章 データとは何か\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert len(prefaces) == 1

    def test_あとがきが検出される(self) -> None:
        pages = [
            _make_page(0, "# 第1章 データとは何か\n本文..."),
            _make_page(1, "# あとがき\n終わりに"),
        ]
        chapters = detect_via_body_headings(pages)
        afterwords = [c for c in chapters if c.kind == ChapterKind.AFTERWORD]
        assert len(afterwords) == 1

    def test_おわりにがAFTERWORDとして検出される(self) -> None:
        pages = [
            _make_page(0, "# 第1章 データとは何か\n本文..."),
            _make_page(1, "# おわりに\n終わりに"),
        ]
        chapters = detect_via_body_headings(pages)
        afterwords = [c for c in chapters if c.kind == ChapterKind.AFTERWORD]
        assert len(afterwords) == 1

    def test_索引が検出される(self) -> None:
        pages = [
            _make_page(0, "# 第1章 データとは何か\n本文..."),
            _make_page(1, "# 索引\nあ行..."),
        ]
        chapters = detect_via_body_headings(pages)
        indexes = [c for c in chapters if c.kind == ChapterKind.INDEX]
        assert len(indexes) == 1

    def test_前書きのtitle_はまえがき(self) -> None:
        pages = [_make_page(0, "# まえがき\n本書について")]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert prefaces[0].title == "まえがき"

    def test_前書きのchapter_numberはNone(self) -> None:
        pages = [_make_page(0, "# まえがき\n本書について")]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert prefaces[0].chapter_number is None


class TestFail_Fast:
    def test_章見出しが1つも検出されない場合はエラー(self) -> None:
        pages = [
            _make_page(0, "## 小見出し\n本文..."),
            _make_page(1, "続き"),
        ]
        with pytest.raises(ChapterDetectionError):
            detect_via_body_headings(pages)

    def test_空ページリストはエラー(self) -> None:
        with pytest.raises(ChapterDetectionError):
            detect_via_body_headings([])
