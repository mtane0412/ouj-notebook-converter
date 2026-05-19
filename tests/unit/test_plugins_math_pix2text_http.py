"""仕様: Pix2TextHttpDetector の HTTP リクエスト・レスポンス処理の動作検証。"""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from ouj_notebook_converter.plugins.math.base import FormulaDetection, MathEngineError
from ouj_notebook_converter.plugins.math.pix2text_http import Pix2TextHttpDetector


@pytest.fixture()
def detector() -> Pix2TextHttpDetector:
    return Pix2TextHttpDetector(base_url="http://localhost:8503", timeout_sec=60.0)


@pytest.fixture()
def dummy_png(tmp_path: Path) -> Path:
    png_path = tmp_path / "数式ページ.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return png_path


class TestPix2TextHttpDetector:
    def test_detect_and_recognizeはbase_urlのdetectエンドポイントへ画像をPOSTする(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_post = mocker.patch("httpx.post", return_value=mock_response)

        detector.detect_and_recognize(dummy_png)

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:8503/detect"

    def test_requestにmultipartのimageフィールドが含まれる(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_post = mocker.patch("httpx.post", return_value=mock_response)

        detector.detect_and_recognize(dummy_png)

        call_kwargs = mock_post.call_args[1]
        assert "files" in call_kwargs
        assert "image" in call_kwargs["files"]

    def test_レスポンスJSONがFormulaDetectionのリストとして返る(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"box": [10, 20, 100, 80], "type": "isolated", "latex": r"\frac{1}{2}", "score": 0.95},
            {"box": [5, 100, 200, 140], "type": "embedding", "latex": r"x^2", "score": 0.88},
        ]
        mocker.patch("httpx.post", return_value=mock_response)

        result = detector.detect_and_recognize(dummy_png)

        assert len(result) == 2
        assert isinstance(result[0], FormulaDetection)
        assert result[0].box == (10, 20, 100, 80)
        assert result[0].type == "isolated"
        assert result[0].latex == r"\frac{1}{2}"
        assert result[0].score == pytest.approx(0.95)
        assert result[1].box == (5, 100, 200, 140)
        assert result[1].type == "embedding"

    def test_レスポンスのlatexがstripされる(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"box": [0, 0, 50, 50], "type": "isolated", "latex": "  \\alpha  \n", "score": 0.9},
        ]
        mocker.patch("httpx.post", return_value=mock_response)

        result = detector.detect_and_recognize(dummy_png)

        assert result[0].latex == r"\alpha"

    def test_typeとscoreとboxが正しく型変換される(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # box は float が返ってくる場合もある（Pix2Text の実装依存）
        mock_response.json.return_value = [
            {"box": [1.7, 2.3, 99.9, 49.1], "type": "embedding", "latex": "y", "score": 0.75},
        ]
        mocker.patch("httpx.post", return_value=mock_response)

        result = detector.detect_and_recognize(dummy_png)

        det = result[0]
        assert all(isinstance(c, int) for c in det.box)
        assert isinstance(det.score, float)
        assert det.box == (1, 2, 99, 49)

    def test_HTTPステータス非2xxならMathEngineErrorを送出(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mocker.patch("httpx.post", return_value=mock_response)

        with pytest.raises(MathEngineError, match="Pix2Text API がエラーを返しました"):
            detector.detect_and_recognize(dummy_png)

    def test_httpx接続失敗ならMathEngineErrorを送出(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("httpx.post", side_effect=httpx.ConnectError("接続失敗"))

        with pytest.raises(MathEngineError, match="Pix2Text API への接続に失敗"):
            detector.detect_and_recognize(dummy_png)

    def test_タイムアウトでMathEngineErrorを送出(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(MathEngineError, match="Pix2Text API がタイムアウトしました"):
            detector.detect_and_recognize(dummy_png)

    def test_不正なJSONレスポンスならMathEngineErrorを送出(
        self, detector: Pix2TextHttpDetector, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("JSON parse error")
        mock_response.text = "not json"
        mocker.patch("httpx.post", return_value=mock_response)

        with pytest.raises(MathEngineError, match="Pix2Text API レスポンスをパースできません"):
            detector.detect_and_recognize(dummy_png)
