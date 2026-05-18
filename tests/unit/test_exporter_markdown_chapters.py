"""仕様: export_markdown_by_chapters の単体テスト。

章ごとに Markdown ファイルを出力し、ファイル名・assets コピー・
画像パス書き換えが正しく動作することを検証する。
"""
from __future__ import annotations

from pathlib import Path

from ouj_notebook_converter.exporters.markdown import export_markdown_by_chapters
from ouj_notebook_converter.pipeline.types import ChapterKind, ChapterSpec, PageMarkdown


def _make_page(
    page_index: int,
    markdown: str = "テスト本文",
    figure_paths: list[Path] | None = None,
) -> PageMarkdown:
    return PageMarkdown(
        page_index=page_index,
        markdown=markdown,
        referenced_assets=figure_paths or [],
    )


def _make_chapter(
    order: int,
    kind: ChapterKind,
    title: str,
    start: int,
    end: int,
    chapter_number: int | None = None,
) -> ChapterSpec:
    return ChapterSpec(
        order=order,
        kind=kind,
        chapter_number=chapter_number,
        title=title,
        start_page_index=start,
        end_page_index=end,
        source="body_headings",
    )


class Test章分割ファイル生成:
    def test_章ごとに別ファイルが作成される(self, tmp_path: Path) -> None:
        pages = [_make_page(i) for i in range(5)]
        chapters = [
            _make_chapter(0, ChapterKind.PREFACE, "まえがき", 0, 0),
            _make_chapter(1, ChapterKind.CHAPTER, "データとは何か", 1, 3, chapter_number=1),
            _make_chapter(2, ChapterKind.AFTERWORD, "あとがき", 4, 4),
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        output_files = list(tmp_path.glob("*.md"))
        assert len(output_files) == 3

    def test_ファイル名はゼロパディング番号_kind形式(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [
            _make_chapter(0, ChapterKind.PREFACE, "まえがき", 0, 0),
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert any(f.startswith("00_preface") for f in file_names)

    def test_章ファイル名にタイトルのスラグが含まれる(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [
            _make_chapter(1, ChapterKind.CHAPTER, "データとは何か", 0, 0, chapter_number=1),
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert any("データとは何か" in f for f in file_names)

    def test_前書きはprefix_preface(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [_make_chapter(0, ChapterKind.PREFACE, "まえがき", 0, 0)]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert any("preface" in f for f in file_names)

    def test_後書きはprefix_afterword(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [_make_chapter(0, ChapterKind.AFTERWORD, "あとがき", 0, 0)]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert any("afterword" in f for f in file_names)

    def test_索引はprefix_index(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [_make_chapter(0, ChapterKind.INDEX, "索引", 0, 0)]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert any("index" in f for f in file_names)

    def test_章ファイルには対応ページの本文が含まれる(self, tmp_path: Path) -> None:
        pages = [
            _make_page(0, "ページ0の本文"),
            _make_page(1, "ページ1の本文"),
            _make_page(2, "ページ2の本文"),
        ]
        chapters = [
            _make_chapter(0, ChapterKind.CHAPTER, "第1章", 0, 1, chapter_number=1),
            _make_chapter(1, ChapterKind.CHAPTER, "第2章", 2, 2, chapter_number=2),
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        chapter1_files = sorted(tmp_path.glob("00_chapter_*.md"))
        assert len(chapter1_files) == 1
        content = chapter1_files[0].read_text(encoding="utf-8")
        assert "ページ0の本文" in content
        assert "ページ1の本文" in content
        assert "ページ2の本文" not in content

    def test_assetsファイルが正しくコピーされる(self, tmp_path: Path) -> None:
        figure = tmp_path / "fig_001.png"
        figure.write_bytes(b"\x89PNG")

        pages = [_make_page(0, "![図](fig_001.png)", figure_paths=[figure])]
        chapters = [_make_chapter(0, ChapterKind.CHAPTER, "第1章", 0, 0, chapter_number=1)]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        assert (assets_dir / "fig_001.png").exists()

    def test_戻り値は書き出したファイルパスのリスト(self, tmp_path: Path) -> None:
        pages = [_make_page(0), _make_page(1)]
        chapters = [
            _make_chapter(0, ChapterKind.PREFACE, "まえがき", 0, 0),
            _make_chapter(1, ChapterKind.CHAPTER, "第1章", 1, 1, chapter_number=1),
        ]
        assets_dir = tmp_path / "assets"

        result = export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert all(p.exists() for p in result)


class Testスラグ変換:
    def test_スラグ化_禁止文字は_に置換される(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [
            _make_chapter(
                1, ChapterKind.CHAPTER, "データ/クレンジング:処理", 0, 0, chapter_number=1
            )
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert all("/" not in f and ":" not in f for f in file_names)

    def test_スラグ化_全角空白は_に置換される(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [
            _make_chapter(
                1, ChapterKind.CHAPTER, "データ　クレンジング", 0, 0, chapter_number=1
            )
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert all("　" not in f for f in file_names)

    def test_スラグ化_連続アンダースコアは1つに圧縮される(self, tmp_path: Path) -> None:
        pages = [_make_page(0)]
        chapters = [
            _make_chapter(
                1, ChapterKind.CHAPTER, "データ//クレンジング", 0, 0, chapter_number=1
            )
        ]
        assets_dir = tmp_path / "assets"

        export_markdown_by_chapters(pages, chapters, tmp_path, assets_dir)

        file_names = [f.name for f in tmp_path.glob("*.md")]
        assert all("__" not in f for f in file_names)
