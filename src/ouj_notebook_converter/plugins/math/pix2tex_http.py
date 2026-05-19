"""仕様: 別 venv で起動中の pix2tex API サーバーへ画像を POST して LaTeX を取得するエンジン。

pix2tex API サーバーのエンドポイント: POST /predict/
リクエスト: multipart/form-data, フィールド名 "file"
レスポンス: LaTeX 文字列（plain text）

事前準備（ユーザー側）:
    # 別 venv にて
    pip install "pix2tex[api]"
    python -m pix2tex.api.run --host 0.0.0.0 --port 8502
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from ouj_notebook_converter.plugins.math.base import MathEngineError


@dataclass(frozen=True)
class Pix2TexHttpEngine:
    """pix2tex HTTP API サーバーへ画像を POST して LaTeX を取得する数式エンジン。"""

    base_url: str = "http://localhost:8502"
    timeout_sec: float = 30.0

    def recognize(self, image_path: Path) -> str:
        """PNG 画像を pix2tex API に送信して LaTeX 文字列を返す。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            LaTeX ソース文字列（先頭末尾の空白を除去済み）。

        Raises:
            MathEngineError: HTTP エラー・接続失敗・タイムアウトの場合。
        """
        url = f"{self.base_url}/predict/"
        try:
            with image_path.open("rb") as f:
                response = httpx.post(
                    url,
                    files={"file": f},
                    timeout=self.timeout_sec,
                )
        except httpx.ConnectError as e:
            raise MathEngineError(
                f"pix2tex API への接続に失敗しました: {url}\n"
                "pix2tex API サーバーが起動していることを確認してください。"
            ) from e
        except httpx.TimeoutException as e:
            raise MathEngineError(
                f"pix2tex API がタイムアウトしました ({self.timeout_sec}秒): {url}"
            ) from e

        if response.status_code >= 300:
            raise MathEngineError(
                f"pix2tex API がエラーを返しました: {response.status_code} {response.text!r}"
            )

        return response.text.strip()
