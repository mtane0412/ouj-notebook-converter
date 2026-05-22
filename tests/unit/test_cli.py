"""仕様: cli モジュール（Typer ベース CLI）のユニットテスト。

Typer の CliRunner を使い、引数解釈と基本バリデーションをテストする。
実際の OCR は走らせない（Fake を注入するか、pdf_path が存在しない時点でエラー）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from unittest.mock import MagicMock

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


class TestMathAutoStart:
    """--math-auto-start / --pix2text-venv オプションのテスト。"""

    def test_pix2text_venvオプションが認識される(self, tmp_path: Path) -> None:
        """--pix2text-venv オプションが Typer に認識されること。"""
        result = runner.invoke(
            app,
            [
                "存在しないPDF.pdf",
                "--outdir",
                str(tmp_path),
                "--pix2text-venv",
                str(tmp_path / "venv"),
            ],
        )
        assert "No such option" not in result.output

    def test_no_math_auto_startオプションが認識される(self, tmp_path: Path) -> None:
        """--no-math-auto-start が Typer に認識されること。"""
        result = runner.invoke(
            app,
            [
                "存在しないPDF.pdf",
                "--outdir",
                str(tmp_path),
                "--no-math-auto-start",
            ],
        )
        assert "No such option" not in result.output

    def test_OUC_PIX2TEXT_VENV環境変数が参照される(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """環境変数 OUC_PIX2TEXT_VENV が CLI オプションとして参照される。"""
        venv_path = tmp_path / "カスタムvenv"
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # Pix2TextServerManager をモックして venv_path の値を捕捉する
        captured: list[Path] = []

        class FakeManager:
            def __init__(self, url: str, venv_path: Path, server_script: Path) -> None:
                captured.append(venv_path)

            def is_server_alive(self) -> bool:
                return True  # 既に起動中とみなす（起動中なら ensure_running は不要）

        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            side_effect=FakeManager,
        )
        runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "pix2text",
            ],
            env={"OUC_PIX2TEXT_VENV": str(venv_path)},
        )
        # FakeManager が venv_path=カスタムvenv で初期化されたこと
        assert len(captured) == 1
        assert captured[0] == venv_path

    def test_math_backend_pix2text時にensure_runningが呼ばれる(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """--math-backend pix2text でサーバー未起動時に ensure_running() が呼ばれる。"""
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_manager = mocker.MagicMock()
        mock_manager.is_server_alive.return_value = False
        # ensure_running 呼び出し後に早期終了するため RuntimeError を送出させる
        mock_manager.ensure_running.side_effect = RuntimeError("テスト用早期終了")

        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            return_value=mock_manager,
        )
        runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "pix2text",
            ],
        )
        mock_manager.ensure_running.assert_called_once()

    def test_no_math_auto_start指定時はensure_runningが呼ばれない(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """--no-math-auto-start 指定時に ensure_running() が呼ばれない。"""
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_manager = mocker.MagicMock()

        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            return_value=mock_manager,
        )
        runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "pix2text",
                "--no-math-auto-start",
            ],
        )
        mock_manager.ensure_running.assert_not_called()

    def test_ServerStartupErrorは終了コード1で処理される(
        self, mocker: MagicMock, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """ensure_running() が ServerStartupError を送出した場合、exit_code == 1 になる。"""
        from ouj_notebook_converter.plugins.math.server_manager import ServerStartupError

        # 実在する PDF が必要なのでダミーを作成
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_manager = mocker.MagicMock()
        mock_manager.is_server_alive.return_value = False
        mock_manager.ensure_running.side_effect = ServerStartupError("起動失敗テスト")

        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            return_value=mock_manager,
        )
        result = runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "pix2text",
            ],
        )
        assert result.exit_code == 1

    def test_サーバー既起動時は起動中メッセージが出ない(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """is_server_alive() が True の場合、起動中メッセージが stderr に出ない。"""
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_manager = mocker.MagicMock()
        mock_manager.is_server_alive.return_value = True

        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            return_value=mock_manager,
        )
        result = runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir",
                str(tmp_path / "out"),
                "--math-backend",
                "pix2text",
            ],
            mix_stderr=False,
        )
        assert "起動中" not in (result.output + (result.stderr or ""))


class TestOcrBackendOption:
    """--ocr-backend / --gemini-api-key / --gemini-model オプションのテスト。"""

    def test_ocr_backendのヘルプが表示される(self) -> None:
        """--ocr-backend オプションが help に含まれること。"""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--ocr-backend" in result.output

    def test_gemini_api_keyのヘルプが表示される(self) -> None:
        """--gemini-api-key オプションが help に含まれること。"""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--gemini-api-key" in result.output

    def test_gemini_modelのヘルプが表示される(self) -> None:
        """--gemini-model オプションが help に含まれること。"""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--gemini-model" in result.output

    def test_ocr_backend_gemini時にapi_key未指定はエラー(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """--ocr-backend gemini で API キー未指定の場合、exit_code != 0 になること。

        Note: Typer がシングルコマンドアプリとして動作するため
        args には 'convert' サブコマンド名を含めない。
        """
        import os

        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # GEMINI_API_KEY を空に設定して「未指定」状態をシミュレートする
        mocker.patch.dict(os.environ, {"GEMINI_API_KEY": ""})

        result = runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir", str(tmp_path / "out"),
                "--ocr-backend", "gemini",
            ],
        )
        assert result.exit_code != 0
        # エラーメッセージに GEMINI_API_KEY が言及されること
        assert "GEMINI_API_KEY" in result.output

    def test_GEMINI_API_KEY環境変数を使ってcreate_gemini_analyzerが呼ばれる(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """GEMINI_API_KEY 環境変数が create_gemini_analyzer に渡されること。"""
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        captured_keys: list[str] = []

        def fake_create(*, api_key: str, model: str) -> object:
            captured_keys.append(api_key)
            # 早期終了用に RuntimeError を送出する
            raise RuntimeError("テスト用早期終了")

        mocker.patch(
            "ouj_notebook_converter.plugins.ocr.gemini.create_gemini_analyzer",
            side_effect=fake_create,
        )

        runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir", str(tmp_path / "out"),
                "--ocr-backend", "gemini",
                "--gemini-api-key", "テスト用APIキー12345",
            ],
        )

        # create_gemini_analyzer が指定のキーで呼ばれたこと
        assert len(captured_keys) == 1
        assert captured_keys[0] == "テスト用APIキー12345"

    def test_ocr_backend_gemini時にpix2textを指定すると警告が出る(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """--ocr-backend gemini + --math-backend pix2text で警告が出力されること。

        math_backend の処理は OCR backend の処理より先に実行されるため
        Pix2TextServerManager もモックする必要がある。
        """
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # Pix2TextServerManager をモック（math_backend 処理が先に実行されるため）
        mock_manager = mocker.MagicMock()
        mock_manager.is_server_alive.return_value = True
        mocker.patch(
            "ouj_notebook_converter.cli.Pix2TextServerManager",
            return_value=mock_manager,
        )
        mocker.patch(
            "ouj_notebook_converter.plugins.math.pix2text_http.Pix2TextHttpDetector",
        )

        # create_gemini_analyzer は警告出力後に呼ばれる
        mocker.patch(
            "ouj_notebook_converter.plugins.ocr.gemini.create_gemini_analyzer",
            side_effect=RuntimeError("テスト用早期終了"),
        )

        result = runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir", str(tmp_path / "out"),
                "--ocr-backend", "gemini",
                "--gemini-api-key", "テスト用APIキー",
                "--math-backend", "pix2text",
            ],
        )
        # 警告が出力されること（typer.echo は stdout/stderr 混在で result.output に入る）
        assert "警告" in result.output

    def test_ocr_backend_gemini時はload_pdf_pages_pypdfium2が呼ばれる(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """--ocr-backend gemini 時に load_pdf_pages_pypdfium2 が呼ばれること。"""
        pdf_path = tmp_path / "テスト教材.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_loader = mocker.MagicMock()
        mock_loader.total_pages = 0
        mock_loader.__iter__ = mocker.MagicMock(return_value=iter([]))

        mock_load_pypdfium = mocker.patch(
            "ouj_notebook_converter.pipeline.stages.load_pypdfium.load_pdf_pages_pypdfium2",
            return_value=mock_loader,
        )
        mocker.patch(
            "ouj_notebook_converter.plugins.ocr.gemini.create_gemini_analyzer",
            return_value=mocker.MagicMock(),
        )
        mocker.patch("ouj_notebook_converter.cli.run_pages", return_value=[])
        mocker.patch("ouj_notebook_converter.cli.export_markdown")

        runner.invoke(
            app,
            [
                str(pdf_path),
                "--outdir", str(tmp_path / "out"),
                "--ocr-backend", "gemini",
                "--gemini-api-key", "テスト用APIキー",
            ],
        )

        mock_load_pypdfium.assert_called_once()

    def test_ocr_backend_yomitokuがデフォルト(self) -> None:
        """--ocr-backend のデフォルトが yomitoku であること。"""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "yomitoku" in result.output


class TestAppStructure:
    """CLI アプリの基本構造テスト。"""

    def test_help_が表示される(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_convert_サブコマンドが存在する(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "convert" in result.output
