"""仕様: CLI の --split / --course-code オプションの単体テスト。

--split chapters 指定時に章分割エクスポーターが呼ばれること、
ChapterDetectionError 時は終了コード 2 になることを検証する。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ouj_notebook_converter.cli import app

_RUNNER = CliRunner()
_MODULE_CLI = "ouj_notebook_converter.cli"
_MODULE_DETECT = "ouj_notebook_converter.pipeline.stages.chapter_detect"


def _make_dummy_pdf(tmp_path: Path) -> Path:
    """テスト用ダミー PDF ファイルを作成する。"""
    pdf = tmp_path / "テスト教科書.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return pdf


class TestSplitオプション:
    def test_splitオプションのデフォルトはnone(self, tmp_path: Path) -> None:
        """--split を省略した場合は既存の combine 動作になる。"""
        pdf = _make_dummy_pdf(tmp_path)
        with (
            patch(f"{_MODULE_CLI}.run_pages", return_value=[]),
            patch(f"{_MODULE_CLI}.load_pdf_pages") as mock_load,
            patch(f"{_MODULE_CLI}.create_analyzer", return_value=MagicMock()),
            patch(f"{_MODULE_CLI}.export_markdown") as mock_export,
        ):
            mock_loader = MagicMock()
            mock_loader.total_pages = 1
            mock_load.return_value = mock_loader

            result = _RUNNER.invoke(app, [
                str(pdf), "-o", str(tmp_path), "--no-cache",
            ])

        # エラーなし（exit code 0）かつ章分割エクスポーターは呼ばれない
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()

    def test_split_chaptersが指定されると章分割エクスポーターが呼ばれる(
        self, tmp_path: Path
    ) -> None:
        from ouj_notebook_converter.pipeline.types import ChapterKind, ChapterSpec, PageMarkdown

        pdf = _make_dummy_pdf(tmp_path)
        fake_pages = [PageMarkdown(page_index=0, markdown="テスト")]
        fake_chapters = [
            ChapterSpec(
                order=1,
                kind=ChapterKind.CHAPTER,
                chapter_number=1,
                title="テスト章",
                start_page_index=0,
                end_page_index=0,
                source="body_headings",
            )
        ]
        with (
            patch(f"{_MODULE_CLI}.run_pages", return_value=fake_pages),
            patch(f"{_MODULE_CLI}.load_pdf_pages") as mock_load,
            patch(f"{_MODULE_CLI}.create_analyzer", return_value=MagicMock()),
            patch(
                f"{_MODULE_CLI}.detect_chapters", return_value=fake_chapters
            ) as mock_detect,
            patch(f"{_MODULE_CLI}.export_markdown_by_chapters") as mock_export_chapters,
        ):
            mock_loader = MagicMock()
            mock_loader.total_pages = 1
            mock_load.return_value = mock_loader

            result = _RUNNER.invoke(app, [
                str(pdf), "-o", str(tmp_path),
                "--split", "chapters", "--no-cache",
            ])

        assert result.exit_code == 0, result.output
        mock_detect.assert_called_once()
        mock_export_chapters.assert_called_once()

    def test_split_chaptersでChapterDetectionError時は終了コード2(
        self, tmp_path: Path
    ) -> None:
        from ouj_notebook_converter.pipeline.stages.chapter_detect import ChapterDetectionError
        from ouj_notebook_converter.pipeline.types import PageMarkdown

        pdf = _make_dummy_pdf(tmp_path)
        fake_pages = [PageMarkdown(page_index=0, markdown="テスト")]
        with (
            patch(f"{_MODULE_CLI}.run_pages", return_value=fake_pages),
            patch(f"{_MODULE_CLI}.load_pdf_pages") as mock_load,
            patch(f"{_MODULE_CLI}.create_analyzer", return_value=MagicMock()),
            patch(
                f"{_MODULE_CLI}.detect_chapters",
                side_effect=ChapterDetectionError("章検出失敗"),
            ),
        ):
            mock_loader = MagicMock()
            mock_loader.total_pages = 1
            mock_load.return_value = mock_loader

            result = _RUNNER.invoke(app, [
                str(pdf), "-o", str(tmp_path),
                "--split", "chapters", "--no-cache",
            ])

        assert result.exit_code == 2

    def test_course_codeオプションが章検出器に渡される(self, tmp_path: Path) -> None:
        from ouj_notebook_converter.pipeline.types import ChapterKind, ChapterSpec, PageMarkdown

        pdf = _make_dummy_pdf(tmp_path)
        fake_pages = [PageMarkdown(page_index=0, markdown="テスト")]
        fake_chapters = [
            ChapterSpec(
                order=1,
                kind=ChapterKind.CHAPTER,
                chapter_number=1,
                title="テスト章",
                start_page_index=0,
                end_page_index=0,
                source="syllabus",
            )
        ]
        with (
            patch(f"{_MODULE_CLI}.run_pages", return_value=fake_pages),
            patch(f"{_MODULE_CLI}.load_pdf_pages") as mock_load,
            patch(f"{_MODULE_CLI}.create_analyzer", return_value=MagicMock()),
            patch(
                f"{_MODULE_CLI}.detect_chapters", return_value=fake_chapters
            ) as mock_detect,
            patch(f"{_MODULE_CLI}.export_markdown_by_chapters"),
        ):
            mock_loader = MagicMock()
            mock_loader.total_pages = 1
            mock_load.return_value = mock_loader

            _RUNNER.invoke(app, [
                str(pdf), "-o", str(tmp_path),
                "--split", "chapters", "--course-code", "1234567", "--no-cache",
            ])

        # detect_chapters に course_code が渡されることを確認する
        call_kwargs = mock_detect.call_args.kwargs
        assert call_kwargs.get("course_code") == "1234567"
