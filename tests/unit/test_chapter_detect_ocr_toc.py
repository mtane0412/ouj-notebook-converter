"""仕様: detect_via_ocr_toc の単体テスト。

OCR 済みの目次ページ Markdown から章タイトルとページ番号を抽出し、
章境界を正しく計算できることを検証する。
"""

from __future__ import annotations

import pytest

from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_via_ocr_toc,
)
from ouj_notebook_converter.pipeline.types import ChapterKind, PageMarkdown


def _make_page(page_index: int, markdown: str = "") -> PageMarkdown:
    return PageMarkdown(page_index=page_index, markdown=markdown)


# 典型的な目次ページの Markdown
_TYPICAL_TOC_MD = """\
# 目次

第 1 章 データとは何か ............ 10
第 2 章 データの収集と前処理 ........ 30
第 3 章 統計的分析の基礎 ........... 55
第 4 章 機械学習入門 .............. 80
"""

_TYPICAL_BODY_PAGES = [
    _make_page(0, "# まえがき\n本書について"),
    _make_page(1, "# 目次\n第1章..."),  # 目次ページ（ダミー）
    _make_page(9, "# 第1章 データとは何か\n本文"),
    _make_page(29, "# 第2章 データの収集と前処理\n本文"),
    _make_page(54, "# 第3章 統計的分析の基礎\n本文"),
    _make_page(79, "# 第4章 機械学習入門\n本文"),
    _make_page(99, "# あとがき\n終わり"),
]


class TestDetectViaOcrToc:
    def test_目次ページからエントリを抽出できる(self) -> None:
        pages = [
            _make_page(0, "# まえがき\n本書について"),
            _make_page(1, _TYPICAL_TOC_MD),
            *[_make_page(i + 2, "本文") for i in range(98)],
        ]
        # 本文中の第1章見出しを用意してオフセット推定を安定させる
        pages[10] = _make_page(10, "# 第1章 データとは何か\n本文")
        pages[30] = _make_page(30, "# 第2章 データの収集と前処理\n本文")

        chapters = detect_via_ocr_toc(pages)
        chapter_items = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_items) >= 2

    def test_章タイトルが抽出される(self) -> None:
        pages = [
            _make_page(0, _TYPICAL_TOC_MD),
            *[_make_page(i + 1, "本文") for i in range(99)],
        ]
        pages[10] = _make_page(10, "# 第1章 データとは何か\n本文")

        chapters = detect_via_ocr_toc(pages)
        chapter_items = sorted(
            [c for c in chapters if c.kind == ChapterKind.CHAPTER],
            key=lambda c: c.chapter_number or 0,
        )
        assert chapter_items[0].title == "データとは何か"

    def test_目次ページが先頭15ページ外にある場合はエラー(self) -> None:
        pages = [
            *[_make_page(i, "本文") for i in range(15)],
            _make_page(15, _TYPICAL_TOC_MD),  # 16ページ目（0-origin:15）
        ]
        with pytest.raises(ChapterDetectionError, match="目次ページ"):
            detect_via_ocr_toc(pages)

    def test_目次ページが見つからない場合はエラー(self) -> None:
        pages = [_make_page(i, "# 第1章 データとは何か\n本文") for i in range(5)]
        with pytest.raises(ChapterDetectionError, match="目次ページ"):
            detect_via_ocr_toc(pages)

    def test_目次エントリが抽出できない場合はエラー(self) -> None:
        pages = [
            _make_page(0, "# 目次\n目次の内容がOCRで読み取れなかった"),
        ]
        with pytest.raises(ChapterDetectionError):
            detect_via_ocr_toc(pages)

    def test_sourceはocr_toc(self) -> None:
        pages = [
            _make_page(0, _TYPICAL_TOC_MD),
            *[_make_page(i + 1, "本文") for i in range(20)],
        ]
        pages[10] = _make_page(10, "# 第1章 データとは何か\n本文")

        chapters = detect_via_ocr_toc(pages)
        assert all(c.source == "ocr_toc" for c in chapters)


# OUJ 実書籍形式の目次ページ (# 目次 見出しなし、# NTitle 形式で章を列挙)
_TOC_PAGE_NTITLE_1 = """\
# まえがき3

# 1Rと RStudio の基本操作

1. R と RStudio9

2. R の基本操作10

# 2R を用いた行列の計算

1. 記述統計量25

# 3ファイルの読み込みとデータフレーム 43

1. データフレームとファイルの読み込み43
"""

_TOC_PAGE_NTITLE_2 = """\
# 4データの視覚化

1. 散布図60

# 5確率分布

1. 確率分布81

60

81
"""


class TestDetectViaOcrTocNTitle形式:
    """OUJ 実書籍形式: # 目次 見出しなし、# NTitle 形式で章を列挙する目次。"""

    def test_NTitle形式のTOCページから章を検出できる(self) -> None:
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            _make_page(1, _TOC_PAGE_NTITLE_2),
            *[_make_page(i + 2, "本文") for i in range(100)],
        ]
        chapters = detect_via_ocr_toc(pages)
        numbers = sorted(c.chapter_number for c in chapters if c.kind == ChapterKind.CHAPTER)
        assert 1 in numbers
        assert 2 in numbers
        assert 3 in numbers
        assert 4 in numbers
        assert 5 in numbers

    def test_NTitle形式から章タイトルが取得される(self) -> None:
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            *[_make_page(i + 1, "本文") for i in range(50)],
        ]
        chapters = detect_via_ocr_toc(pages)
        ch1 = next((c for c in chapters if c.chapter_number == 1), None)
        assert ch1 is not None
        assert ch1.title == "Rと RStudio の基本操作"

    def test_NTitle形式から章開始ページが取得される(self) -> None:
        """# NTitle 形式の目次: 最初の節エントリのページ番号が章開始ページになる。"""
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            *[_make_page(i + 1, "本文") for i in range(50)],
        ]
        chapters = detect_via_ocr_toc(pages)
        ch1 = next((c for c in chapters if c.chapter_number == 1), None)
        assert ch1 is not None
        assert ch1.start_page_index == 9  # 節エントリ「R と RStudio9」のページ番号

    def test_NTitle形式で見出し末尾のページ番号が使われる(self) -> None:
        """# 3ファイルの読み込みとデータフレーム 43 のように見出し末尾に番号がある場合。"""
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            *[_make_page(i + 1, "本文") for i in range(50)],
        ]
        chapters = detect_via_ocr_toc(pages)
        ch3 = next((c for c in chapters if c.chapter_number == 3), None)
        assert ch3 is not None
        assert ch3.start_page_index == 43  # 見出し末尾の「43」

    def test_NTitle形式でまえがきが検出される(self) -> None:
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            *[_make_page(i + 1, "本文") for i in range(50)],
        ]
        chapters = detect_via_ocr_toc(pages)
        prefaces = [c for c in chapters if c.kind == ChapterKind.PREFACE]
        assert len(prefaces) == 1
        assert prefaces[0].start_page_index == 3  # まえがき3 → ページ3

    def test_NTitle形式のsourceはocr_toc(self) -> None:
        pages = [
            _make_page(0, _TOC_PAGE_NTITLE_1),
            *[_make_page(i + 1, "本文") for i in range(50)],
        ]
        chapters = detect_via_ocr_toc(pages)
        assert all(c.source == "ocr_toc" for c in chapters)
