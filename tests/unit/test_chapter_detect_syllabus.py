"""仕様: detect_via_syllabus の単体テスト。

httpx.Client をモック化し、フィクスチャ HTML を返すオフライン専用テスト。
実際の OUJ サイトへのアクセスは行わない。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_via_syllabus,
)
from ouj_notebook_converter.pipeline.types import ChapterKind, PageMarkdown


def _make_page(page_index: int, markdown: str = "") -> PageMarkdown:
    return PageMarkdown(page_index=page_index, markdown=markdown)


# テスト用シラバス HTML（OUJ の実際の構造に類似した簡易フィクスチャ）
_SYLLABUS_HTML = """\
<!DOCTYPE html>
<html>
<body>
<table>
  <tr><th>回</th><th>内容</th></tr>
  <tr><td>第 1 章 データとは何か</td><td>データの基本概念を学ぶ</td></tr>
  <tr><td>第 2 章 データの収集と前処理</td><td>データ収集方法を学ぶ</td></tr>
  <tr><td>第 3 章 統計的分析の基礎</td><td>統計的手法を学ぶ</td></tr>
</table>
</body>
</html>
"""

# 本文ページ（章見出しを持つページ群）
_BODY_PAGES = [
    _make_page(0, "# まえがき\n本書について"),
    _make_page(1, "# 第1章 データとは何か\n本文"),
    _make_page(2, "# 第2章 データの収集と前処理\n本文"),
    _make_page(3, "# 第3章 統計的分析の基礎\n本文"),
    _make_page(4, "# あとがき\n終わりに"),
]


def _make_mock_client(html: str, status_code: int = 200) -> MagicMock:
    """httpx.Client のモックを作成する。"""
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_response,
        )

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    return mock_client


class TestDetectViaSyllabus:
    def test_シラバスHTMLから章タイトルを抽出できる(self) -> None:
        mock_client = _make_mock_client(_SYLLABUS_HTML)
        chapters = detect_via_syllabus("1554069", _BODY_PAGES, mock_client)

        chapter_items = [c for c in chapters if c.kind == ChapterKind.CHAPTER]
        assert len(chapter_items) == 3

    def test_シラバスのタイトルが章に反映される(self) -> None:
        mock_client = _make_mock_client(_SYLLABUS_HTML)
        chapters = detect_via_syllabus("1554069", _BODY_PAGES, mock_client)

        chapter_items = sorted(
            [c for c in chapters if c.kind == ChapterKind.CHAPTER],
            key=lambda c: c.chapter_number or 0,
        )
        assert chapter_items[0].title == "データとは何か"
        assert chapter_items[1].title == "データの収集と前処理"

    def test_正しいURLでHTTPリクエストが送信される(self) -> None:
        mock_client = _make_mock_client(_SYLLABUS_HTML)
        detect_via_syllabus("1234567", _BODY_PAGES, mock_client)

        call_args = mock_client.get.call_args
        assert "1234567" in call_args[0][0]

    def test_HTTPエラー時はChapterDetectionErrorを送出する(self) -> None:
        mock_client = _make_mock_client("", status_code=404)
        with pytest.raises(ChapterDetectionError, match="シラバス取得に失敗"):
            detect_via_syllabus("9999999", _BODY_PAGES, mock_client)

    def test_HTMLから章タイトルが抽出できない場合はエラー(self) -> None:
        empty_html = "<html><body><p>章タイトルが存在しないHTML</p></body></html>"
        mock_client = _make_mock_client(empty_html)
        with pytest.raises(ChapterDetectionError, match="章タイトルを抽出できません"):
            detect_via_syllabus("1554069", _BODY_PAGES, mock_client)

    def test_sourceはsyllabus(self) -> None:
        mock_client = _make_mock_client(_SYLLABUS_HTML)
        chapters = detect_via_syllabus("1554069", _BODY_PAGES, mock_client)
        assert all(c.source == "syllabus" for c in chapters)
