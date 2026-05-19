"""仕様: scripts/pix2text_server.py の /health・/detect・/recognize エンドポイントの動作検証。

pix2text 本体は別 venv（~/.venvs/pix2text）で動作するため、
build_app() に fake な MathFormulaDetector / LatexOCR を注入してテストする。
モジュールレベルの pix2text インポートは sys.modules でモックして解消する。
"""

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# pix2text は別 venv のため、モジュール全体をモックしてインポートを通す
_pix2text_mock = MagicMock()
sys.modules.setdefault("pix2text", _pix2text_mock)


@pytest.fixture()
def dummy_png_bytes() -> bytes:
    """1x1 ピクセルの白色 PNG バイト列を返す。"""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class FakeMathFormulaDetector:
    """テスト用の固定検出結果を返す MathFormulaDetector 代替。"""

    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self._detections = detections or []

    def detect(self, np_img: np.ndarray) -> list[dict[str, Any]]:
        return self._detections


class FakeLatexOCR:
    """テスト用の固定 LaTeX を返す LatexOCR 代替。"""

    def __init__(self, latex: str = r"\frac{1}{2}", score: float = 0.95) -> None:
        self._latex = latex
        self._score = score
        self.called_images: list[Image.Image] = []

    def recognize(self, pil_img: Image.Image) -> dict[str, Any]:
        self.called_images.append(pil_img)
        return {"text": self._latex, "score": self._score}


class TestHealthEndpoint:
    def test_healthはokをTrueで返す(self, dummy_png_bytes: bytes) -> None:
        from fastapi.testclient import TestClient

        from scripts.pix2text_server import build_app

        app = build_app(FakeMathFormulaDetector(), FakeLatexOCR())
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


class TestRecognizeEndpoint:
    def test_recognizeは与えられた画像をLatexOCRに渡しlatexとscoreを返す(
        self, dummy_png_bytes: bytes
    ) -> None:
        from fastapi.testclient import TestClient

        from scripts.pix2text_server import build_app

        fake_mfr = FakeLatexOCR(latex=r"\alpha", score=0.88)
        app = build_app(FakeMathFormulaDetector(), fake_mfr)
        client = TestClient(app)

        response = client.post("/recognize", files={"image": ("数式.png", dummy_png_bytes, "image/png")})

        assert response.status_code == 200
        data = response.json()
        assert data["latex"] == r"\alpha"
        assert data["score"] == pytest.approx(0.88)

    def test_recognizeはFakeLatexOCRのrecognizeを1回呼ぶ(
        self, dummy_png_bytes: bytes
    ) -> None:
        from fastapi.testclient import TestClient

        from scripts.pix2text_server import build_app

        fake_mfr = FakeLatexOCR(latex=r"x^2")
        app = build_app(FakeMathFormulaDetector(), fake_mfr)
        client = TestClient(app)

        client.post("/recognize", files={"image": ("数式.png", dummy_png_bytes, "image/png")})

        assert len(fake_mfr.called_images) == 1

    def test_recognizeはlatexが空白でも正常に返す(
        self, dummy_png_bytes: bytes
    ) -> None:
        from fastapi.testclient import TestClient

        from scripts.pix2text_server import build_app

        fake_mfr = FakeLatexOCR(latex="", score=0.1)
        app = build_app(FakeMathFormulaDetector(), fake_mfr)
        client = TestClient(app)

        response = client.post("/recognize", files={"image": ("空.png", dummy_png_bytes, "image/png")})

        assert response.status_code == 200
        assert response.json()["latex"] == ""
