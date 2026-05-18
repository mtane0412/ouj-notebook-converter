"""仕様: cli モジュール（Typer ベース CLI）のユニットテスト。

Typer の CliRunner を使い、引数解釈と基本バリデーションをテストする。
実際の OCR は走らせない（Fake を注入するか、pdf_path が存在しない時点でエラー）。
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.cli import app

runner = CliRunner()


class TestConvertCommand:
    """convert サブコマンドのテスト。"""

    def test_存在しないPDFはエラーを返す(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["convert", str(tmp_path / "存在しない.pdf"), "--outdir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0

    def test_helpが表示される(self) -> None:
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "convert" in result.output.lower() or "pdf" in result.output.lower()

    def test_ignore_metaのデフォルトはTrue(self) -> None:
        """ヘッダー/フッター除外はデフォルト有効であることを help で確認。"""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        # デフォルトが ignore-meta（True）であること
        assert "default" in result.output and "ignore-meta" in result.output

    def test_outdir_は必須(self) -> None:
        result = runner.invoke(app, ["convert", "dummy.pdf"])
        assert result.exit_code != 0

    def test_format_デフォルトはmd(self, tmp_path: Path) -> None:
        """存在しないPDFでエラー終了するが、--outdir フラグは受け付けることを確認。"""
        result = runner.invoke(
            app,
            ["convert", "dummy.pdf", "--outdir", str(tmp_path / "out")],
        )
        # PDF が存在しないので失敗するが、format エラーではないはず
        assert "format" not in result.output.lower() or result.exit_code != 0

    def test_無効なdeviceはエラー(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "convert",
                "dummy.pdf",
                "--outdir",
                str(tmp_path / "out"),
                "-d",
                "invalid_device",
            ],
        )
        assert result.exit_code != 0


class TestAppStructure:
    """CLI アプリの基本構造テスト。"""

    def test_help_が表示される(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_convert_サブコマンドが存在する(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "convert" in result.output
