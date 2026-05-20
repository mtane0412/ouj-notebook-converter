"""仕様: exporters.epub モジュール（EPUB エクスポーター）のユニットテスト。

PageMarkdown のリストから EPUB3 ファイルを生成する処理をテストする。
EPUB は ZIP アーカイブ形式であり、テキストが検索可能な形式で格納される。
pandoc が未インストールの環境ではテストをスキップする。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ouj_notebook_converter.exporters.epub import export_epub
from ouj_notebook_converter.pipeline.types import PageMarkdown

PANDOC_AVAILABLE = shutil.which("pandoc") is not None


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


class TestExportEpubValidation:
    """入力バリデーションのテスト（pandoc 不要）。"""

    def test_pagesが空の場合はValueErrorを送出する(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pages が空"):
            export_epub([], tmp_path / "out.epub", tmp_path / "assets")

    def test_pypandoc未インストール時はImportErrorを送出する(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        with patch.dict("sys.modules", {"pypandoc": None}):
            with pytest.raises(ImportError):
                export_epub(pages, tmp_path / "out.epub", tmp_path / "assets")


@pytest.mark.skipif(not PANDOC_AVAILABLE, reason="pandoc がインストールされていません")
class TestExportEpub:
    """EPUB 生成のテスト（pandoc 必須）。"""

    def test_出力EPUBファイルが作成される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        out_epub = tmp_path / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets")
        assert out_epub.exists()

    def test_EPUBはZIPアーカイブ形式である(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        out_epub = tmp_path / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets")
        # EPUB はZIP形式（先頭バイトがPKシグネチャ）
        assert zipfile.is_zipfile(out_epub)

    def test_EPUBのmimetypeエントリが正しい(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "# 第1章\n本文テキスト")]
        out_epub = tmp_path / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets")
        with zipfile.ZipFile(out_epub) as zf:
            assert "mimetype" in zf.namelist()
            assert zf.read("mimetype") == b"application/epub+zip"

    def test_複数ページの内容がEPUBに含まれる(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "最初のページの内容"),
            _make_page(1, "二番目のページの内容"),
        ]
        out_epub = tmp_path / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets")
        # ZIP 内のテキストファイルに内容が含まれることを確認
        with zipfile.ZipFile(out_epub) as zf:
            all_content = "\n".join(
                zf.read(name).decode("utf-8", errors="ignore")
                for name in zf.namelist()
                if name.endswith(".html") or name.endswith(".xhtml")
            )
        assert "最初のページ" in all_content
        assert "二番目のページ" in all_content

    def test_titleが指定された場合はメタデータに反映される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "本文テキスト")]
        out_epub = tmp_path / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets", title="放送大学テキスト")
        # タイトルが ZIP 内のいずれかのファイルに含まれることを確認
        with zipfile.ZipFile(out_epub) as zf:
            all_content = "\n".join(
                zf.read(name).decode("utf-8", errors="ignore") for name in zf.namelist()
            )
        assert "放送大学テキスト" in all_content

    def test_出力ディレクトリが存在しなくても作成される(self, tmp_path: Path) -> None:
        pages = [_make_page(0, "テスト")]
        out_epub = tmp_path / "subdir" / "output.epub"
        export_epub(pages, out_epub, tmp_path / "assets")
        assert out_epub.exists()

    def test_figureが埋め込まれる(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        fig = cache_dir / "fig_001.png"
        # 最小 PNG（1×1ピクセルの白画像）
        fig.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        pages = [_make_page(0, f"本文\n\n![図1]({fig})\n", figure_paths=[fig])]
        out_epub = tmp_path / "output.epub"
        assets_dir = tmp_path / "assets"
        export_epub(pages, out_epub, assets_dir)
        with zipfile.ZipFile(out_epub) as zf:
            names = zf.namelist()
        # pandoc は画像を EPUB/media/ 以下に配置してリネームするため、PNG が存在することを確認する
        assert any(name.endswith(".png") for name in names)
