"""仕様: utils.pages モジュール（ページ範囲パース）のユニットテスト。"""

import pytest

from ouj_notebook_converter.utils.pages import parse_page_range


class TestParsePageRange:
    """parse_page_range 関数のテスト。"""

    def test_単一ページ(self) -> None:
        assert parse_page_range("3", total=10) == [3]

    def test_連続範囲(self) -> None:
        assert parse_page_range("2-5", total=10) == [2, 3, 4, 5]

    def test_カンマ区切り(self) -> None:
        assert parse_page_range("1,3,7", total=10) == [1, 3, 7]

    def test_混在指定(self) -> None:
        assert parse_page_range("1,3-5,10", total=10) == [1, 3, 4, 5, 10]

    def test_重複は除去される(self) -> None:
        assert parse_page_range("1-3,2-4", total=10) == [1, 2, 3, 4]

    def test_ソートされて返る(self) -> None:
        assert parse_page_range("5,1,3", total=10) == [1, 3, 5]

    def test_Noneを渡すと全ページを返す(self) -> None:
        assert parse_page_range(None, total=5) == [1, 2, 3, 4, 5]

    def test_空文字列はエラー(self) -> None:
        with pytest.raises(ValueError, match="ページ範囲"):
            parse_page_range("", total=10)

    def test_totalを超えるページはエラー(self) -> None:
        with pytest.raises(ValueError, match="total"):
            parse_page_range("1-15", total=10)

    def test_0以下はエラー(self) -> None:
        with pytest.raises(ValueError, match="1以上"):
            parse_page_range("0", total=10)

    def test_範囲の開始が終了より大きいとエラー(self) -> None:
        with pytest.raises(ValueError, match="範囲"):
            parse_page_range("5-3", total=10)
