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


class Test数字直結形式の章:
    """OUJ 実書籍の OCR 出力は「# 1Rと RStudio の基本操作」形式を使う。"""

    def test_数字直結形式の章が検出される(self) -> None:
        """OUJ 実書籍形式: # 1Rと RStudio の基本操作 → CHAPTER"""
        pages = [
            _make_page(0, "# まえがき3\n本書について"),
            _make_page(1, "# 1Rと RStudio の基本操作\n本文..."),
            _make_page(2, "続き"),
            _make_page(3, "# 2R を用いた行列の計算\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_chapters) == 2

    def test_数字直結形式の章番号が取得される(self) -> None:
        pages = [_make_page(0, "# 1Rと RStudio の基本操作\n本文...")]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].chapter_number == 1

    def test_数字直結形式の2桁章番号が取得される(self) -> None:
        pages = [_make_page(0, "# 15データの分類\n本文...")]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].chapter_number == 15

    def test_数字直結形式の章タイトルから数字プレフィックスが除去される(self) -> None:
        pages = [_make_page(0, "# 1Rと RStudio の基本操作\n本文...")]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].title == "Rと RStudio の基本操作"

    def test_2桁数字直結形式の章タイトルから数字プレフィックスが除去される(self) -> None:
        pages = [_make_page(0, "# 15データの分類\n本文...")]
        chapters = detect_via_body_headings(pages)
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert chapter_chapters[0].title == "データの分類"

    def test_ピリオド区切り見出しは章として検出しない(self) -> None:
        """# 1. セクション形式はセクション見出しであり章ではない。"""
        pages = [_make_page(0, "# 1. セクションタイトル\n本文...")]
        with pytest.raises(ChapterDetectionError):
            detect_via_body_headings(pages)


class Testまえがき末尾数字:
    """OUJ 実書籍の OCR では # まえがき3 のようにページ番号が混入する。"""

    def test_まえがきに末尾数字が付いていてもPREFACEとして検出される(self) -> None:
        pages = [
            _make_page(0, "# まえがき3\n本書について"),
            _make_page(1, "# 1Rと RStudio の基本操作\n本文..."),
        ]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert len(prefaces) == 1

    def test_まえがきタイトルから末尾数字が除去される(self) -> None:
        pages = [_make_page(0, "# まえがき3\n本書について")]
        chapters = detect_via_body_headings(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert prefaces[0].title == "まえがき"


class Test目次ページの除外:
    """複数の H1 見出しを持つページ（目次ページ）は章境界として検出しない。"""

    def test_複数H1を持つページはスキップされる(self) -> None:
        """OUJ 実書籍の目次ページ: 1ページに複数の章見出しが列挙される。"""
        toc_page = (
            "# 4データの視覚化\n\n"
            "1. 散布図60\n\n"
            "# 5確率分布\n\n"
            "1. 確率分布81\n\n"
            "# 6アソシエーション分析\n\n"
            "1. POS システム102\n"
        )
        pages = [
            _make_page(0, toc_page),
            _make_page(1, "# 4データの視覚化\n《目標&ポイント》..."),
        ]
        chapters = detect_via_body_headings(pages)
        # 目次ページ(p0)は除外され、実際の第4章ページ(p1)のみ検出される
        chapter_chapters = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_chapters) == 1
        assert chapter_chapters[0].start_page_index == 1

    def test_括弧付き番号見出しは章として検出しない(self) -> None:
        """# 4\\) 図の見出し\\(タイトル\\) はセクション見出しであり章ではない。"""
        pages = [
            _make_page(0, "# 4\\) 図の見出し\\(タイトル\\)\n本文..."),
        ]
        with pytest.raises(ChapterDetectionError):
            detect_via_body_headings(pages)
