"""仕様: 自前の FastAPI ラッパー (scripts/pix2text_server.py) へ画像を POST して数式を検出・認識するエンジン。

エンドポイント: POST /detect
リクエスト   : multipart/form-data, フィールド名 "image"
レスポンス   : [{"box":[x1,y1,x2,y2], "type":"isolated"|"embedding", "latex":str, "score":float}, ...]

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

    def detect_and_recognize(self, image_path: Path) -> list[FormulaDetection]:
        """PNG 画像を Pix2Text API に送信して数式検出・認識結果を返す。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            検出された数式ごとの FormulaDetection リスト。数式なしの場合は空リスト。

        Raises:
            MathEngineError: HTTP エラー・接続失敗・タイムアウト・JSON パース失敗の場合。
        """
        url = f"{self.base_url}/detect"
        try:
            with image_path.open("rb") as f:
                response = httpx.post(
                    url,
                    files={"image": f},
                    timeout=self.timeout_sec,
                )
        except httpx.ConnectError as e:
            raise MathEngineError(
                f"Pix2Text API への接続に失敗しました: {url}\n"
                "scripts/pix2text_server.py が起動していることを確認してください。"
            ) from e
        except httpx.TimeoutException as e:
            raise MathEngineError(
                f"Pix2Text API がタイムアウトしました ({self.timeout_sec}秒): {url}"
            ) from e

        if response.status_code >= 300:
            raise MathEngineError(
                f"Pix2Text API がエラーを返しました: {response.status_code} {response.text!r}"
            )

        try:
            raw_list: list[dict[str, Any]] = response.json()
        except (ValueError, KeyError) as e:
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
