"""仕様: Pix2Text HTTP サーバーのプロセスライフサイクル管理モジュール。

--math-backend pix2text 指定時に ~/.venvs/pix2text 仮想環境の
scripts/pix2text_server.py を subprocess で自動起動し、atexit で自動終了する。

起動フロー:
  1. is_server_alive() で既存サーバーの死活を確認
  2. 起動していなければ start() → _wait_for_ready() を実行
  3. プロセス終了時に atexit 経由で stop() が呼ばれる

エラー:
  - venv/bin/python が存在しない場合 → ServerStartupError
  - タイムアウト（デフォルト 60 秒）以内に起動しない場合 → ServerStartupError
"""

from __future__ import annotations

import atexit
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_STARTUP_TIMEOUT_SEC: float = 60.0
DEFAULT_POLL_INTERVAL_SEC: float = 1.0


class ServerStartupError(RuntimeError):
    """Pix2Text サーバーの起動失敗を示す例外。"""


@dataclass
class Pix2TextServerManager:
    """Pix2Text HTTP サーバーのプロセスを管理するクラス。

    Attributes:
        url: サーバーの URL（例: "http://127.0.0.1:8503"）。
        venv_path: pix2text をインストールした venv のルートパス。
        server_script: pix2text_server.py のパス。
        startup_timeout_sec: サーバー起動待機タイムアウト（秒）。
        poll_interval_sec: ヘルスチェックのポーリング間隔（秒）。
    """

    url: str
    venv_path: Path
    server_script: Path
    startup_timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def is_server_alive(self) -> bool:
        """GET /health を呼び出してサーバーが起動中かを確認する。

        Returns:
            200 かつ {"ok": true} の場合 True、それ以外は False。
        """
        try:
            response = httpx.get(f"{self.url}/health", timeout=2.0)
            return response.status_code == 200 and response.json().get("ok") is True
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    def start(self) -> None:
        """venv の Python で pix2text_server.py をサブプロセスとして起動する。

        atexit に stop() を登録するため、プロセス終了時に自動でサーバーが停止される。

        Raises:
            ServerStartupError: venv/bin/python が存在しない場合。
        """
        python_bin = self.venv_path / "bin" / "python"
        if not python_bin.exists():
            raise ServerStartupError(
                f"Python バイナリが見つかりません: {python_bin}\n"
                f"~/.venvs/pix2text に pix2text[serve] をインストールしてください:\n"
                f"  python3 -m venv ~/.venvs/pix2text\n"
                f"  ~/.venvs/pix2text/bin/pip install 'pix2text[serve]'"
            )

        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8503)

        self._process = subprocess.Popen(
            [str(python_bin), str(self.server_script), "--host", host, "--port", port],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.stop)

    def _wait_for_ready(self) -> None:
        """サーバーが応答を返すまでポーリングする。

        Raises:
            ServerStartupError: タイムアウト以内に起動しなかった場合。
        """
        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self.is_server_alive():
                return
            time.sleep(self.poll_interval_sec)

        self.stop()
        raise ServerStartupError(
            f"Pix2Text サーバーが {self.startup_timeout_sec:.0f} 秒以内に起動しませんでした。\n"
            f"手動で起動してから --no-math-auto-start を指定してください:\n"
            f"  {self.venv_path}/bin/python {self.server_script}"
        )

    def ensure_running(self) -> None:
        """サーバーが起動していなければ起動し、レディネスを待つ。

        既に起動中の場合は何もしない。
        """
        if self.is_server_alive():
            return
        self.start()
        self._wait_for_ready()

    def stop(self) -> None:
        """サーバープロセスを終了する。

        _process が None の場合は何もしない（冪等）。
        """
        if self._process is None:
            return
        self._process.terminate()
        self._process.wait()
        self._process = None
