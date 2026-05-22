"""仕様: Gemini OCR バックエンドのユニットテスト。

GeminiAnalyzerResult と GeminiAnalyzer の動作を検証する。
Gemini API は pytest-mock でモックするため、実際の API キーは不要。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# GeminiAnalyzerResult のテスト
# ---------------------------------------------------------------------------


class TestGeminiAnalyzerResult:
    """GeminiAnalyzerResult（AnalyzerResult プロトコル実装）の動作検証。"""

    def _get_result(self) -> object:
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        return GeminiAnalyzerResult("# 放送大学テキスト\n\n本文テキスト")

    def test_to_jsonはbackendフィールドにgeminiを書き出す(self, tmp_path: Path) -> None:
        """to_json が {"backend": "gemini", ...} の JSON を書き出すこと。"""
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        result = GeminiAnalyzerResult("テストMarkdown")
        out_path = tmp_path / "analysis.json"

        result.to_json(out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["backend"] == "gemini"

    def test_to_jsonはmarkdownテキストを含む(self, tmp_path: Path) -> None:
        """to_json の JSON に markdown テキストが含まれること。"""
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        result = GeminiAnalyzerResult("# 見出し\n\n本文")
        out_path = tmp_path / "analysis.json"

        result.to_json(out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["markdown"] == "# 見出し\n\n本文"

    def test_to_jsonはparagraphsとwordsに空リストを含む(self, tmp_path: Path) -> None:
        """to_json が paragraphs と words を空リストで書き出すこと。

        math_detect.py が KeyError を起こさないための措置。
        """
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        result = GeminiAnalyzerResult("テスト")
        out_path = tmp_path / "analysis.json"

        result.to_json(out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["paragraphs"] == []
        assert data["words"] == []

    def test_to_markdownはmarkdownテキストをそのままファイルに書き出す(
        self, tmp_path: Path
    ) -> None:
        """to_markdown が Gemini の出力 Markdown をそのままファイルに書き出すこと。"""
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        expected = "# 経済学の基礎\n\n第1章 市場の仕組み"
        result = GeminiAnalyzerResult(expected)
        out_path = tmp_path / "raw.md"

        result.to_markdown(out_path)

        assert out_path.read_text(encoding="utf-8") == expected

    def test_to_markdownはimgキーワードを受け取っても無視する(
        self, tmp_path: Path
    ) -> None:
        """to_markdown が img kwarg を受け取ってもエラーにならないこと。"""
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        result = GeminiAnalyzerResult("テスト")
        out_path = tmp_path / "raw.md"
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # img を渡してもエラーにならないこと
        result.to_markdown(out_path, img=dummy_img)

        assert out_path.exists()

    def test_to_markdownはignore_line_breakキーワードを受け取っても無視する(
        self, tmp_path: Path
    ) -> None:
        """to_markdown が ignore_line_break kwarg を受け取ってもエラーにならないこと。"""
        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        result = GeminiAnalyzerResult("テスト")
        out_path = tmp_path / "raw.md"

        result.to_markdown(out_path, ignore_line_break=True)

        assert out_path.exists()


# ---------------------------------------------------------------------------
# GeminiAnalyzer のテスト
# ---------------------------------------------------------------------------


def _build_mock_genai(response_text: str = "# OCR結果\n\nテキスト") -> MagicMock:
    """google.genai モジュールのモックを構築するヘルパー。"""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    return mock_genai


class TestGeminiAnalyzer:
    """GeminiAnalyzer（AnalyzerProtocol 実装）の動作検証。"""

    def _make_analyzer(self, mock_genai: MagicMock) -> object:
        """モックを差し込んで GeminiAnalyzer を生成するヘルパー。"""
        mock_types = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            from importlib import reload

            # モジュールをリロードして import を再実行させる
            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            reload(gemini_module)
            analyzer = gemini_module.GeminiAnalyzer(api_key="テスト用APIキー")
        return analyzer, gemini_module

    def test_callはタプルを返す(self) -> None:
        """__call__ が (GeminiAnalyzerResult, None, None) のタプルを返すこと。"""
        mock_genai = _build_mock_genai("# テスト")
        mock_types = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.GeminiAnalyzer(api_key="テスト用APIキー")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result, vis1, vis2 = analyzer(image)

        from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult

        assert isinstance(result, GeminiAnalyzerResult)
        assert vis1 is None
        assert vis2 is None

    def test_APIレスポンスのtextがresultのmarkdownになる(self) -> None:
        """Gemini API のレスポンステキストが結果の Markdown テキストになること。"""
        expected_text = "# 数学的分析\n\n式: $E = mc^2$"
        mock_genai = _build_mock_genai(expected_text)
        mock_types = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.GeminiAnalyzer(api_key="テスト用APIキー")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result, _, _ = analyzer(image)

        # to_markdown の結果が API レスポンスと一致すること
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            result.to_markdown(tmp_path)
            assert tmp_path.read_text(encoding="utf-8") == expected_text
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_BGRndarrayをJPEG変換してAPIを呼ぶ(self) -> None:
        """BGR ndarray が JPEG bytes に変換されて Gemini API に渡されること。"""
        mock_genai = _build_mock_genai()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.GeminiAnalyzer(api_key="テスト用APIキー")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            analyzer(image)

            # from google.genai import types → mock_genai.types が使われる
            # types.Part.from_bytes が mime_type="image/jpeg" で呼ばれること
            assert mock_genai.types.Part.from_bytes.called
            call_kwargs = mock_genai.types.Part.from_bytes.call_args
            assert call_kwargs.kwargs.get("mime_type") == "image/jpeg"

    def test_APIエラー時はRuntimeErrorを送出する(self) -> None:
        """Gemini API 呼び出しが失敗した場合に RuntimeError が送出されること。"""
        mock_genai = _build_mock_genai()
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.side_effect = Exception("APIエラー: 認証失敗")
        mock_types = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.GeminiAnalyzer(api_key="テスト用APIキー")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            with pytest.raises(RuntimeError, match="Gemini API"):
                analyzer(image)


# ---------------------------------------------------------------------------
# create_gemini_analyzer のテスト
# ---------------------------------------------------------------------------


class TestCreateGeminiAnalyzer:
    """create_gemini_analyzer ファクトリ関数の動作検証。"""

    def test_google_genai未インストール時はImportError(self) -> None:
        """google-genai 未インストール時に ImportError が送出されること（Fail-Fast）。"""
        # google.genai を None で上書きしてインストールされていない状態を模倣
        with patch.dict(
            sys.modules,
            {
                "google": None,  # type: ignore[dict-item]
                "google.genai": None,  # type: ignore[dict-item]
                "google.genai.types": None,  # type: ignore[dict-item]
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)

            with pytest.raises(ImportError, match="google-genai"):
                gemini_module.create_gemini_analyzer(api_key="テスト用APIキー")

    def test_正常にGeminiAnalyzerを返す(self) -> None:
        """google-genai がインストールされている場合に GeminiAnalyzer を返すこと。"""
        mock_genai = _build_mock_genai()
        mock_types = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.create_gemini_analyzer(api_key="テスト用APIキー")

        assert isinstance(analyzer, gemini_module.GeminiAnalyzer)

    def test_デフォルトモデルはgemini_3_5_flash(self) -> None:
        """デフォルトモデルが gemini-3.5-flash であること。"""
        mock_genai = _build_mock_genai()
        mock_types = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            import importlib

            import ouj_notebook_converter.plugins.ocr.gemini as gemini_module

            importlib.reload(gemini_module)
            analyzer = gemini_module.create_gemini_analyzer(api_key="テスト用APIキー")

        assert analyzer._model == "gemini-3.5-flash"
