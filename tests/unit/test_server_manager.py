"""仕様: Pix2TextServerManager のユニットテスト。

subprocess と httpx をモックして、サーバー起動・ヘルスチェック・
停止の各メソッドが正しく振る舞うことを検証する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ouj_notebook_converter.plugins.math.server_manager import (
    Pix2TextServerManager,
    ServerStartupError,
)


def _make_manager(tmp_path: Path) -> tuple[Pix2TextServerManager, Path]:
    """テスト用の Pix2TextServerManager とダミー python_bin を返す。"""
    venv_path = tmp_path / "venv"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.touch()

    manager = Pix2TextServerManager(
        url="http://127.0.0.1:8503",
        venv_path=venv_path,
        server_script=Path("scripts/pix2text_server.py"),
        startup_timeout_sec=0.5,
        poll_interval_sec=0.05,
    )
    return manager, python_bin


class TestIsServerAlive:
    """is_server_alive() のテスト。"""

    def test_healthエンドポイントが200かつok_TrueならTrueを返す(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """GET /health が {"ok": true} を返す場合 True になる。"""
        manager, _ = _make_manager(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mocker.patch("httpx.get", return_value=mock_response)

        assert manager.is_server_alive() is True

    def test_接続失敗ならFalseを返す(self, mocker: MagicMock, tmp_path: Path) -> None:
        """httpx.ConnectError が発生した場合 False になる。"""
        import httpx

        manager, _ = _make_manager(tmp_path)
        mocker.patch("httpx.get", side_effect=httpx.ConnectError("接続失敗"))

        assert manager.is_server_alive() is False

    def test_タイムアウトならFalseを返す(self, mocker: MagicMock, tmp_path: Path) -> None:
        """httpx.TimeoutException が発生した場合 False になる。"""
        import httpx

        manager, _ = _make_manager(tmp_path)
        mocker.patch("httpx.get", side_effect=httpx.TimeoutException("タイムアウト"))

        assert manager.is_server_alive() is False

    def test_okがFalseのレスポンスならFalseを返す(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """{"ok": false} の場合 False になる。"""
        manager, _ = _make_manager(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False}
        mocker.patch("httpx.get", return_value=mock_response)

        assert manager.is_server_alive() is False

    def test_非2xxステータスならFalseを返す(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """HTTP 500 の場合 False になる。"""
        manager, _ = _make_manager(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}
        mocker.patch("httpx.get", return_value=mock_response)

        assert manager.is_server_alive() is False


class TestStart:
    """start() のテスト。"""

    def test_venvのpythonが存在しない場合ServerStartupErrorを送出(
        self, tmp_path: Path
    ) -> None:
        """venv_path/bin/python が存在しない場合、ServerStartupError が発生する。"""
        manager = Pix2TextServerManager(
            url="http://127.0.0.1:8503",
            venv_path=tmp_path / "存在しないvenv",
            server_script=Path("scripts/pix2text_server.py"),
        )
        with pytest.raises(ServerStartupError, match="Python バイナリが見つかりません"):
            manager.start()

    def test_subprocessPopenが正しいコマンドで呼ばれる(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """start() が subprocess.Popen を正しい引数で呼び出す。"""
        manager, python_bin = _make_manager(tmp_path)
        mock_popen = mocker.patch("subprocess.Popen", return_value=MagicMock())
        mocker.patch("atexit.register")

        manager.start()

        mock_popen.assert_called_once_with(
            [str(python_bin), "scripts/pix2text_server.py", "--host", "127.0.0.1", "--port", "8503"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_startはatexitにstopを登録する(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """start() が atexit.register(manager.stop) を呼び出す。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch("subprocess.Popen", return_value=MagicMock())
        mock_atexit = mocker.patch("atexit.register")

        manager.start()

        mock_atexit.assert_called_once_with(manager.stop)


class TestWaitForReady:
    """_wait_for_ready() のテスト。"""

    def test_即座にaliveになれば正常終了(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """is_server_alive() が即 True を返す場合、例外なく完了する。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch.object(manager, "is_server_alive", return_value=True)

        manager._wait_for_ready()  # 例外が起きないこと

    def test_タイムアウトするとServerStartupErrorを送出(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """常に False が返り続ける場合、タイムアウトで ServerStartupError が発生する。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch.object(manager, "is_server_alive", return_value=False)

        with pytest.raises(ServerStartupError, match="起動しませんでした"):
            manager._wait_for_ready()

    def test_数回のポーリング後にaliveになれば正常終了(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """[False, False, True] の順で返る場合、例外なく完了する。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch.object(manager, "is_server_alive", side_effect=[False, False, True])

        manager._wait_for_ready()  # 例外が起きないこと


class TestEnsureRunning:
    """ensure_running() のテスト。"""

    def test_既にaliveならstartを呼ばない(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """is_server_alive() が True の場合、start() は呼ばれない。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch.object(manager, "is_server_alive", return_value=True)
        mock_start = mocker.patch.object(manager, "start")

        manager.ensure_running()

        mock_start.assert_not_called()

    def test_aliveでなければstartとwait_for_readyを呼ぶ(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """is_server_alive() が False の場合、start() と _wait_for_ready() が呼ばれる。"""
        manager, _ = _make_manager(tmp_path)
        mocker.patch.object(manager, "is_server_alive", return_value=False)
        mock_start = mocker.patch.object(manager, "start")
        mock_wait = mocker.patch.object(manager, "_wait_for_ready")

        manager.ensure_running()

        mock_start.assert_called_once()
        mock_wait.assert_called_once()


class TestStop:
    """stop() のテスト。"""

    def test_processが存在する場合terminateとwaitが呼ばれる(
        self, tmp_path: Path
    ) -> None:
        """_process が設定されている場合、terminate() → wait() が呼ばれる。"""
        manager, _ = _make_manager(tmp_path)
        mock_process = MagicMock()
        manager._process = mock_process

        manager.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    def test_processがNoneの場合は何もしない(self, tmp_path: Path) -> None:
        """_process が None の場合、例外が発生しない。"""
        manager, _ = _make_manager(tmp_path)
        # _process は初期状態で None

        manager.stop()  # 例外が起きないこと
