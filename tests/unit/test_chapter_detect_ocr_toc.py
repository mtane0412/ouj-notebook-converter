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
    _make_page(1, "# 目次\n第1章..."),         # 目次ページ（ダミー）
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
