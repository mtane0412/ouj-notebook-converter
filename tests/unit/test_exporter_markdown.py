"""仕様: exporters.markdown モジュール（Markdown エクスポーター）のユニットテスト。

PageMarkdown のリストを受け取り、最終的な Markdown ファイルと
figure アセットを出力先ディレクトリに書き出す処理をテストする。
PageMarkdown.markdown 内の figure 参照はアセットの絶対パスで記述されている前提。
"""
from pathlib import Path

from ouj_notebook_converter.exporters.markdown import export_markdown
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


class TestExportMarkdownCombine:
    """combine=True（デフォルト）のテスト。"""

    def test_単一ページのMarkdownが出力される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        out_md = tmp_path / "output.md"
        assets_dir = tmp_path / "assets"
        export_markdown(pages, out_md, assets_dir)
        assert out_md.exists()
        content = out_md.read_text()
        assert "第1章" in content

    def test_ページ間にページマーカーが挿入される(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "ページ1の内容"),
            _make_page(1, "ページ2の内容"),
        ]
        out_md = tmp_path / "output.md"
        export_markdown(pages, out_md, tmp_path / "assets")
        content = out_md.read_text()
        assert "<!-- page: 1 -->" in content
        assert "<!-- page: 2 -->" in content

    def test_ページの内容が順序通りに結合される(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "最初のページ"),
            _make_page(1, "二番目のページ"),
        ]
        out_md = tmp_path / "output.md"
        export_markdown(pages, out_md, tmp_path / "assets")
        content = out_md.read_text()
        pos1 = content.index("最初のページ")
        pos2 = content.index("二番目のページ")
        assert pos1 < pos2

    def test_figureパスが出力先の相対パスに書き換えられる(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fig = cache_dir / "fig_001.png"
        fig.write_bytes(b"\x89PNG")  # ダミー PNG

        pages = [
            _make_page(
                0,
                f"本文\n\n![図1]({fig})\n\n続き",
                figure_paths=[fig],
            )
        ]
        out_md = tmp_path / "output.md"
        assets_dir = tmp_path / "assets"
        export_markdown(pages, out_md, assets_dir)

        content = out_md.read_text()
        # 絶対パスが残っていないこと
        assert str(cache_dir) not in content
        # アセットがコピーされていること
        assert any(assets_dir.rglob("fig_001.png"))
        # 相対パスに書き換わっていること
        assert "fig_001.png" in content

    def test_figureファイルがassetsディレクトリにコピーされる(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fig = cache_dir / "fig_001.png"
        fig.write_bytes(b"dummy")

        pages = [_make_page(0, f"![図]({fig})", figure_paths=[fig])]
        assets_dir = tmp_path / "assets"
        export_markdown(pages, tmp_path / "out.md", assets_dir)

        assert any(assets_dir.rglob("fig_001.png"))

    def test_出力ディレクトリが存在しなくても作成される(self, tmp_path: Path) -> None:
        out_md = tmp_path / "subdir" / "output.md"
        pages = [_make_page(0, "テスト")]
        export_markdown(pages, out_md, tmp_path / "assets")
        assert out_md.exists()

    def test_figureなしページは正常に処理される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "図なしのページ\n\n本文のみ")]
        out_md = tmp_path / "output.md"
        export_markdown(pages, out_md, tmp_path / "assets")
        content = out_md.read_text()
        assert "図なしのページ" in content


class TestExportMarkdownSeparate:
    """combine=False（ページ個別出力）のテスト。"""

    def test_ページごとに別ファイルが作成される(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "ページ1"),
            _make_page(1, "ページ2"),
        ]
        out_dir = tmp_path / "pages"
        export_markdown(pages, out_dir, tmp_path / "assets", combine=False)
        assert (out_dir / "page_0001.md").exists()
        assert (out_dir / "page_0002.md").exists()

    def test_個別ページにもページマーカーが入る(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "ページ1")]
        out_dir = tmp_path / "pages"
        export_markdown(pages, out_dir, tmp_path / "assets", combine=False)
        content = (out_dir / "page_0001.md").read_text()
        assert "<!-- page: 1 -->" in content
