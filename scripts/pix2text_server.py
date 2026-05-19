"""Pix2Text 自前 HTTP ラッパー。

公式の p2t serve エンドポイントは構造化 JSON を返さないため、
MathFormulaDetector.detect() + LatexOCR.recognize() を直接呼んで
bbox / type / latex / score の JSON を返す薄い FastAPI ラッパー。

## セットアップ（本リポジトリとは別の venv で実行する）

```bash
python3.11 -m venv ~/.venvs/pix2text
~/.venvs/pix2text/bin/pip install "pix2text[serve]"
```

## 起動コマンド

```bash
~/.venvs/pix2text/bin/python scripts/pix2text_server.py --port 8503
```

## エンドポイント

- GET  /health  → {"ok": true}
- POST /detect  multipart/form-data image=<png>
    → [{"box":[x1,y1,x2,y2], "type":"isolated"|"embedding", "latex":str, "score":float}, ...]

## 本体 CLI からの呼び出し

```bash
uv run ounc <PDF> --outdir <dir> --math-backend pix2text --pix2text-url http://localhost:8503
```
"""

from __future__ import annotations

import argparse
import io
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pix2text import LatexOCR, MathFormulaDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_app(mfd: MathFormulaDetector, mfr: LatexOCR) -> FastAPI:
    """FastAPI アプリを構築して返す。

    Args:
        mfd: MathFormulaDetector のインスタンス。
        mfr: LatexOCR のインスタンス。

    Returns:
        設定済みの FastAPI アプリ。
    """
    app = FastAPI(title="Pix2Text Detection Server")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/detect")
    async def detect(image: UploadFile = File(...)) -> list[dict]:
        """ページ画像から数式を検出・認識して JSON を返す。"""
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.asarray(pil_img)

        # MathFormulaDetector.detect() → list[{"box": ndarray(4,2), "type": str, "score": float}]
        raw_detections = mfd.detect(np_img)

        results: list[dict] = []
        for det in raw_detections:
            pts = det["box"]  # shape (4, 2): 4 点の (x, y) 座標
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())

            crop = pil_img.crop((x1, y1, x2, y2))
            # LatexOCR.recognize() はシングル画像の場合 {"text": str, "score": float} を返す
            rec_result = mfr.recognize(crop)
            latex = (rec_result.get("text", "") if isinstance(rec_result, dict) else str(rec_result)).strip()

            results.append(
                {
                    "box": [x1, y1, x2, y2],
                    "type": str(det["type"]),
                    "latex": latex,
                    "score": float(det["score"]),
                }
            )

        logger.info("検出数: %d", len(results))
        return results

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Pix2Text 検出・認識 HTTP サーバー")
    parser.add_argument("--host", default="127.0.0.1", help="バインドホスト")
    parser.add_argument("--port", type=int, default=8503, help="ポート番号")
    args = parser.parse_args()

    logger.info("MathFormulaDetector と LatexOCR を初期化中...")
    mfd = MathFormulaDetector()
    mfr = LatexOCR()
    logger.info("初期化完了。サーバーを起動します: http://%s:%d", args.host, args.port)

    app = build_app(mfd, mfr)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
