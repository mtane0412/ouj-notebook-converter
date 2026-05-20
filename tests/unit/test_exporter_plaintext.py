"""仕様: exporters.plaintext モジュール（TXT エクスポーター）のユニットテスト。

PageMarkdown のリストを受け取り、Markdown 記法を除去したプレーンテキストファイルを
出力先ディレクトリに書き出す処理をテストする。
画像参照 ![alt](path) は [図: alt] に変換され、アセットのコピーは行わない。
"""

from pathlib import Path

import pytest

from ouj_notebook_converter.exporters.plaintext import (
    _markdown_to_plaintext,
    export_plaintext,
)
from ouj_notebook_converter.pipeline.types import PageMarkdown


def _make_page(
    page_index: int,
    markdown: str,
    figure_paths: list[Path] | None = None,
) -> PageMarkdown:
    """テスト用 PageMarkdown を作る簡易ファクトリ。"""
    return PageMarkdown(
        page_index=page_index,
        markdown=markdown,
        referenced_assets=figure_paths or [],
    )


class TestExportPlaintextCombine:
    """combine=True（デフォルト）のテスト。"""

    def test_単一ページのテキストが出力される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        out_txt = tmp_path / "output.txt"
        assets_dir = tmp_path / "assets"
        export_plaintext(pages, out_txt, assets_dir)
        assert out_txt.exists()
        content = out_txt.read_text()
        assert "第1章" in content

    def test_ページの内容が順序通りに結合される(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "最初のページ"),
            _make_page(1, "二番目のページ"),
        ]
        out_txt = tmp_path / "output.txt"
        export_plaintext(pages, out_txt, tmp_path / "assets")
        content = out_txt.read_text()
        pos1 = content.index("最初のページ")
        pos2 = content.index("二番目のページ")
        assert pos1 < pos2

    def test_出力ディレクトリが存在しなくても作成される(self, tmp_path: Path) -> None:
        out_txt = tmp_path / "subdir" / "output.txt"
        pages = [_make_page(0, "テスト")]
        export_plaintext(pages, out_txt, tmp_path / "assets")
        assert out_txt.exists()

    def test_Markdown見出し記号が除去される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n## 第1節\n### 小見出し")]
        out_txt = tmp_path / "output.txt"
        export_plaintext(pages, out_txt, tmp_path / "assets")
        content = out_txt.read_text()
        assert "# " not in content
        assert "## " not in content
        assert "### " not in content
        assert "第1章" in content
        assert "第1節" in content
        assert "小見出し" in content

    def test_画像参照が図テキストに変換される(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fig = cache_dir / "fig_001.png"
        fig.write_bytes(b"\x89PNG")

        pages = [_make_page(0, f"本文\n\n![図1-1]({fig})\n\n続き", figure_paths=[fig])]
        out_txt = tmp_path / "output.txt"
        export_plaintext(pages, out_txt, tmp_path / "assets")
        content = out_txt.read_text()
        # 画像パスが残っていないこと
        assert str(fig) not in content
        # 図の代替テキストが含まれること
        assert "図1-1" in content

    def test_アセットファイルはコピーされない(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fig = cache_dir / "fig_001.png"
        fig.write_bytes(b"dummy")

        pages = [_make_page(0, f"![図]({fig})", figure_paths=[fig])]
        assets_dir = tmp_path / "assets"
        export_plaintext(pages, tmp_path / "out.txt", assets_dir)
        # アセットディレクトリが作成されないか、空であること
        assert not assets_dir.exists() or not any(assets_dir.iterdir())

    def test_pagesが空の場合はValueErrorが発生する(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pages が空"):
            export_plaintext([], tmp_path / "out.txt", tmp_path / "assets")


class TestExportPlaintextSeparate:
    """combine=False（ページ個別出力）のテスト。"""

    def test_ページごとにtxtファイルが作成される(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "ページ1"),
            _make_page(1, "ページ2"),
        ]
        out_dir = tmp_path / "pages"
        export_plaintext(pages, out_dir, tmp_path / "assets", combine=False)
        assert (out_dir / "page_0001.txt").exists()
        assert (out_dir / "page_0002.txt").exists()

    def test_分割ファイルの内容が正しい(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "# 第1ページ見出し\n本文1"),
            _make_page(1, "本文2"),
        ]
        out_dir = tmp_path / "pages"
        export_plaintext(pages, out_dir, tmp_path / "assets", combine=False)
        content1 = (out_dir / "page_0001.txt").read_text()
        assert "第1ページ見出し" in content1
        assert "# " not in content1


class TestMarkdownToPlaintext:
    """_markdown_to_plaintext() の変換ルールの単体テスト。"""

    def test_見出し1が変換される(self) -> None:
        result = _markdown_to_plaintext("# データサイエンスの基礎")
        assert result == "データサイエンスの基礎"

    def test_見出し2が変換される(self) -> None:
        result = _markdown_to_plaintext("## 統計の概要")
        assert result == "統計の概要"

    def test_見出し3が変換される(self) -> None:
        result = _markdown_to_plaintext("### 小見出し")
        assert result == "小見出し"

    def test_アスタリスク太字が変換される(self) -> None:
        result = _markdown_to_plaintext("**重要な概念**です")
        assert result == "重要な概念です"

    def test_アンダースコア太字が変換される(self) -> None:
        result = _markdown_to_plaintext("__重要な概念__です")
        assert result == "重要な概念です"

    def test_アスタリスク斜体が変換される(self) -> None:
        result = _markdown_to_plaintext("*イタリック*テキスト")
        assert result == "イタリックテキスト"

    def test_アンダースコア斜体が変換される(self) -> None:
        result = _markdown_to_plaintext("_斜体_テキスト")
        assert result == "斜体テキスト"

    def test_インラインコードが変換される(self) -> None:
        result = _markdown_to_plaintext("`print()`関数")
        assert result == "print()関数"

    def test_画像参照が図テキストに変換される(self) -> None:
        result = _markdown_to_plaintext("![図1-1](/path/to/fig.png)")
        assert result == "[図: 図1-1]"

    def test_リンクがテキストのみに変換される(self) -> None:
        result = _markdown_to_plaintext("[参考文献](https://example.com)")
        assert result == "参考文献"

    def test_水平線が空行に変換される(self) -> None:
        result = _markdown_to_plaintext("---")
        assert result.strip() == ""

    def test_通常テキストはそのまま(self) -> None:
        result = _markdown_to_plaintext("放送大学のテキスト本文です。")
        assert result == "放送大学のテキスト本文です。"

    def test_空文字列はそのまま(self) -> None:
        result = _markdown_to_plaintext("")
        assert result == ""
