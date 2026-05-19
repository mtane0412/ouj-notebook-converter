"""仕様: 自前の FastAPI ラッパー (scripts/pix2text_server.py) へ画像を POST して数式を検出・認識するエンジン。

エンドポイント:
  POST /detect    multipart/form-data, フィールド名 "image"
                  → [{"box":[x1,y1,x2,y2], "type":"isolated"|"embedding", "latex":str, "score":float}, ...]
  POST /recognize multipart/form-data, フィールド名 "image"
                  → {"latex":str, "score":float}

事前準備（ユーザー側）:
    python3.11 -m venv ~/.venvs/pix2text
    ~/.venvs/pix2text/bin/pip install "pix2text[serve]"
    ~/.venvs/pix2text/bin/python scripts/pix2text_server.py --port 8503
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ouj_notebook_converter.plugins.math.base import FormulaDetection, MathEngineError


@dataclass(frozen=True)
class Pix2TextHttpDetector:
    """Pix2Text 自前 HTTP ラッパーへ画像を POST して数式検出・認識結果を取得するエンジン。"""

    base_url: str = "http://localhost:8503"
    timeout_sec: float = 60.0

    def _post_image(self, endpoint: str, image_path: Path) -> httpx.Response:
        """指定エンドポイントへ画像を multipart POST する共通処理。

        Args:
            endpoint: URL パス（例: "/detect"）。base_url と結合して完全 URL を構成する。
            image_path: 送信する画像ファイルの絶対パス。

        Returns:
            httpx.Response オブジェクト。

        Raises:
            MathEngineError: 接続失敗またはタイムアウトの場合。
        """
        url = f"{self.base_url}{endpoint}"
        try:
            with image_path.open("rb") as f:
                return httpx.post(url, files={"image": f}, timeout=self.timeout_sec)
        except httpx.ConnectError as e:
            raise MathEngineError(
                f"Pix2Text API への接続に失敗しました: {url}\n"
                "scripts/pix2text_server.py が起動していることを確認してください。"
            ) from e
        except httpx.TimeoutException as e:
            raise MathEngineError(
                f"Pix2Text API がタイムアウトしました ({self.timeout_sec}秒): {url}"
            ) from e

    def recognize_image(self, image_path: Path) -> tuple[str, float]:
        """crop 済みの数式画像 1 枚を Pix2Text /recognize API に送って LaTeX を取得する。

        日本語ラベルを除外してトリミングした画像の再認識に使用する。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            (latex, score) のタプル。latex は先頭末尾の空白除去済み。

        Raises:
            MathEngineError: HTTP エラー・接続失敗・タイムアウト・JSON パース失敗の場合。
        """
        response = self._post_image("/recognize", image_path)

        if response.status_code >= 300:
            raise MathEngineError(
                f"Pix2Text API がエラーを返しました: {response.status_code} {response.text!r}"
            )

        try:
            data: dict[str, Any] = response.json()
            return (str(data["latex"]).strip(), float(data["score"]))
        except (TypeError, ValueError, KeyError) as e:
            raise MathEngineError(
                f"Pix2Text API レスポンスをパースできません: {response.text!r}"
            ) from e

    def detect_and_recognize(self, image_path: Path) -> list[FormulaDetection]:
        """PNG 画像を Pix2Text API に送信して数式検出・認識結果を返す。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            検出された数式ごとの FormulaDetection リスト。数式なしの場合は空リスト。

        Raises:
            MathEngineError: HTTP エラー・接続失敗・タイムアウト・JSON パース失敗の場合。
        """
        response = self._post_image("/detect", image_path)

        if response.status_code >= 300:
            raise MathEngineError(
                f"Pix2Text API がエラーを返しました: {response.status_code} {response.text!r}"
            )

        try:
            raw_list: list[dict[str, Any]] = response.json()
        except (TypeError, ValueError, KeyError) as e:
            raise MathEngineError(
                f"Pix2Text API レスポンスをパースできません: {response.text!r}"
            ) from e

        return [
            FormulaDetection(
                box=(
                    int(det["box"][0]),
                    int(det["box"][1]),
                    int(det["box"][2]),
                    int(det["box"][3]),
                ),
                type=str(det["type"]),
                latex=str(det["latex"]).strip(),
                score=float(det["score"]),
            )
            for det in raw_list
        ]
