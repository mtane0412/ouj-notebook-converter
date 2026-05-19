"""仕様: Pix2TexHttpEngine の HTTP リクエスト・レスポンス処理の動作検証。"""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from ouj_notebook_converter.plugins.math.base import MathEngineError
from ouj_notebook_converter.plugins.math.pix2tex_http import Pix2TexHttpEngine


@pytest.fixture()
def engine() -> Pix2TexHttpEngine:
    return Pix2TexHttpEngine(base_url="http://localhost:8502", timeout_sec=30.0)


@pytest.fixture()
def dummy_png(tmp_path: Path) -> Path:
    # 最小の 1x1 PNG（実際の画像内容はテストで不要）
    png_path = tmp_path / "数式.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return png_path


class TestPix2TexHttpEngine:
    def test_recognizeはbase_urlのpredictエンドポイントへPOSTする(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = r"\frac{1}{2}"
        mock_post = mocker.patch("httpx.post", return_value=mock_response)

        engine.recognize(dummy_png)

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:8502/predict/"

    def test_レスポンスのLaTeX文字列が返る(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = r"\frac{a}{b} + \sum_{i=1}^{n}"
        mocker.patch("httpx.post", return_value=mock_response)

        result = engine.recognize(dummy_png)

        assert result == r"\frac{a}{b} + \sum_{i=1}^{n}"

    def test_LaTeXの前後空白がstripされる(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "  \\alpha + \\beta  \n"
        mocker.patch("httpx.post", return_value=mock_response)

        result = engine.recognize(dummy_png)

        assert result == r"\alpha + \beta"

    def test_HTTPステータス非2xxならMathEngineErrorを送出(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mocker.patch("httpx.post", return_value=mock_response)

        with pytest.raises(MathEngineError, match="pix2tex API がエラーを返しました"):
            engine.recognize(dummy_png)

    def test_httpx接続失敗ならMathEngineErrorを送出(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("httpx.post", side_effect=httpx.ConnectError("接続失敗"))

        with pytest.raises(MathEngineError, match="pix2tex API への接続に失敗"):
            engine.recognize(dummy_png)

    def test_タイムアウトでMathEngineErrorを送出(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(MathEngineError, match="pix2tex API がタイムアウトしました"):
            engine.recognize(dummy_png)

    def test_requestにmultipartのfileフィールドが含まれる(
        self, engine: Pix2TexHttpEngine, dummy_png: Path, mocker: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = r"\int x dx"
        mock_post = mocker.patch("httpx.post", return_value=mock_response)

        engine.recognize(dummy_png)

        # files 引数に "file" キーが含まれること
        call_kwargs = mock_post.call_args[1]
        assert "files" in call_kwargs
        assert "file" in call_kwargs["files"]
