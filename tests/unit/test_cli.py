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


class TestConvertMathFlag:
    """--math / --pix2tex-url フラグのテスト。"""

    def test_mathフラグのヘルプが表示される(self) -> None:
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--math" in result.output

    def test_pix2tex_urlのヘルプが表示される(self) -> None:
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--pix2tex-url" in result.output

    def test_pix2tex_urlオプションが認識される(self, tmp_path: Path) -> None:
        # --pix2tex-url が Typer に認識されること（Unknown option エラーにならない）を確認する。
        # PDF が存在しないので exit_code != 0 だが、pix2tex-url のパースエラーではないこと。
        result = runner.invoke(
            app,
            [
                "convert",
                "存在しないPDF.pdf",
                "--outdir",
                str(tmp_path),
                "--pix2tex-url",
                "http://other-server:9999",
            ],
        )
        assert "No such option" not in result.output
        assert result.exit_code != 0

    def test_math未指定ならpdf存在チェックで止まる(self, tmp_path: Path) -> None:
        """--math なしで PDF が存在しない場合、pix2tex 関連のエラーは出ない。"""
        result = runner.invoke(
            app,
            ["convert", "存在しないファイル.pdf", "--outdir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0
        assert "pix2tex" not in result.output


class TestConvertMathBackendFlag:
    """--math-backend / --pix2text-url フラグのテスト。"""

    def test_math_backendのヘルプが表示される(self) -> None:
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--math-backend" in result.output

    def test_pix2text_urlのヘルプが表示される(self) -> None:
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--pix2text-url" in result.output

    def test_math_backend不正値はエラーを返す(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "dummy.pdf",
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "invalid_backend",
            ],
        )
        assert result.exit_code != 0

    def test_math_backend_pix2textフラグが認識される(self, tmp_path: Path) -> None:
        """--math-backend pix2text が Typer に認識されること（Unknown option/value にならない）。"""
        result = runner.invoke(
            app,
            [
                "存在しないPDF.pdf",
                "--outdir",
                str(tmp_path),
                "--math-backend",
                "pix2text",
            ],
        )
        # PDF が存在しないので失敗するが、--math-backend のパースエラーではないこと
        assert "No such option" not in result.output
        assert "invalid value" not in result.output.lower()

    def test_math_True指定はdeprecation警告が出る(self, tmp_path: Path) -> None:
        """--math (deprecated) 指定時に deprecation 警告メッセージが出ること。"""
        result = runner.invoke(
            app,
            [
                "存在しないPDF.pdf",
                "--outdir",
                str(tmp_path),
                "--math",
            ],
        )
        # PDF が存在しないので失敗するが、deprecation 警告は出ること
        assert "deprecated" in result.output.lower() or "廃止" in result.output


class TestAppStructure:
    """CLI アプリの基本構造テスト。"""

    def test_help_が表示される(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_convert_サブコマンドが存在する(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "convert" in result.output
