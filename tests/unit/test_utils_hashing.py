"""仕様: utils.hashing モジュール（ファイルハッシュ算出）のユニットテスト。"""

from pathlib import Path

import pytest

from ouj_notebook_converter.utils.hashing import sha256_file, sha256_string


class TestSha256String:
    """sha256_string 関数のテスト。"""

    def test_同じ文字列は同じハッシュ(self) -> None:
        assert sha256_string("テスト") == sha256_string("テスト")

    def test_異なる文字列は異なるハッシュ(self) -> None:
        assert sha256_string("テスト1") != sha256_string("テスト2")

    def test_64文字の16進数を返す(self) -> None:
        result = sha256_string("テスト")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_空文字列でも動作する(self) -> None:
        result = sha256_string("")
        assert len(result) == 64


class TestSha256File:
    """sha256_file 関数のテスト。"""

    def test_同じ内容のファイルは同じハッシュ(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("放送大学の教科書")
        file_b.write_text("放送大学の教科書")
        assert sha256_file(file_a) == sha256_file(file_b)

    def test_内容が違うと異なるハッシュ(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("データの分析")
        file_b.write_text("問題解決の数理")
        assert sha256_file(file_a) != sha256_file(file_b)

    def test_存在しないファイルはエラー(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "存在しない.pdf")

    def test_8文字の短縮ハッシュを返せる(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("テスト")
        result = sha256_file(f, short=True)
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)
