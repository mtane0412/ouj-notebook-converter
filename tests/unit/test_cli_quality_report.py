"""仕様: CLI の --quality-report オプションの単体テスト。

--quality-report 指定時に変換後の品質レポートが出力されること、
KaTeX を使えない場合は OCR を始める前に終了コード 1 で失敗すること、
指定しない場合は何も出力しないことを検証する。
KaTeX の検査（Node.js 呼び出し）はフェイクに差し替える。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import Result
from typer.testing import CliRunner

from ouj_notebook_converter.cli import app
from ouj_notebook_converter.evaluation.katex import KatexCheckError
from ouj_notebook_converter.evaluation.markdown_parts import Formula
from ouj_notebook_converter.pipeline.types import PageMarkdown

_RUNNER = CliRunner()
_MODULE_CLI = "ouj_notebook_converter.cli"
_MODULE_KATEX = "ouj_notebook_converter.quality_report.check_katex"


def _all_errors(formulas: Sequence[Formula]) -> list[str | None]:
    """すべての数式を描画不能とするフェイク検査。"""
    return ["未定義の命令です" for _ in formulas]


def _invoke(
    tmp_path: Path, extra_args: list[str], check: Callable[..., list[str | None]]
) -> tuple[Result, MagicMock]:
    """OCR とエクスポートをモックして CLI を実行し、結果と run_pages のモックを返す。"""
    pdf = tmp_path / "テスト教科書.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pages = [PageMarkdown(page_index=0, markdown="式 $\\cline{1-2}$ です。")]
    with ExitStack() as stack:
        run_pages = stack.enter_context(patch(f"{_MODULE_CLI}.run_pages", return_value=pages))
        mock_load = stack.enter_context(patch(f"{_MODULE_CLI}.load_pdf_pages"))
        stack.enter_context(patch(f"{_MODULE_CLI}.create_analyzer", return_value=MagicMock()))
        stack.enter_context(patch(f"{_MODULE_CLI}.export_markdown"))
        stack.enter_context(patch(_MODULE_KATEX, side_effect=check))
        mock_load.return_value = MagicMock(total_pages=1)
        result = _RUNNER.invoke(app, [str(pdf), "-o", str(tmp_path), "--no-cache", *extra_args])
    return result, run_pages


class TestQualityReportオプション:
    def test_指定するとレポートが出力される(self, tmp_path: Path) -> None:
        result, _ = _invoke(tmp_path, ["--quality-report"], _all_errors)

        assert result.exit_code == 0, result.output
        assert (tmp_path / "quality_report.json").exists()
        assert (tmp_path / "quality_report.md").exists()
        assert "1 件" in result.output  # 描画できない数式の件数を標準出力にも示す

    def test_KaTeXを使えなければOCR前に終了コード1で失敗する(self, tmp_path: Path) -> None:
        def _broken(formulas: Sequence[Formula]) -> list[str | None]:
            raise KatexCheckError("Node.js（node コマンド）が見つかりません")

        result, run_pages = _invoke(tmp_path, ["--quality-report"], _broken)

        assert result.exit_code == 1
        assert "Node.js" in result.output
        run_pages.assert_not_called()

    def test_指定しなければレポートを出力しない(self, tmp_path: Path) -> None:
        result, _ = _invoke(tmp_path, [], _all_errors)

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "quality_report.json").exists()
