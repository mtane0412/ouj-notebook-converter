"""仕様: Gemini API を使った OCR バックエンド。

AnalyzerProtocol（pipeline/stages/ocr.py）に準拠したアダプターを提供する。
yomitoku が不要なため、GPU/torch 環境なしで OCR を実行できる。

依存: google-genai>=1.0, Pillow>=10.0（pyproject.toml の gemini extra）
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

# OCR 専用プロンプト（日本語教科書ページ向け）
_GEMINI_OCR_PROMPT = (
    "この画像は日本語の教科書ページです。以下の指示に従って OCR を実行してください。\n"
    "1. ページの全テキストを正確に抽出して Markdown 形式で出力する\n"
    "2. 見出しは #/##/### で表現する\n"
    "3. 数式は LaTeX 形式（$...$ または $$...$$）で表現する\n"
    "4. 表は Markdown テーブル形式で表現する\n"
    "5. ページ番号・ヘッダー・フッターは除外する\n"
    "6. テキスト以外の前置き・説明は一切出力しない"
)


class GeminiAnalyzerResult:
    """Gemini OCR 結果を AnalyzerResult プロトコルに適合させるアダプター。

    yomitoku の DocumentAnalyzer が返す results オブジェクトと同じインターフェースを持ち、
    既存パイプライン（analyze_page）にそのまま差し込める。
    """

    def __init__(self, markdown_text: str) -> None:
        self._markdown_text = markdown_text

    def to_json(self, path: str | Path) -> None:
        """最小限の JSON を書き出す。

        math_detect.py が paragraphs/words キーを読むため空リストで書き出す。
        Gemini + pix2text の組み合わせ時に KeyError が起きないための措置。
        """
        data = {
            "backend": "gemini",
            "markdown": self._markdown_text,
            "paragraphs": [],
            "words": [],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self, path: str | Path, **kwargs: Any) -> None:
        """Gemini の Markdown 出力をそのままファイルに書き出す。

        yomitoku 互換のため img / ignore_line_break kwargs を受け取るが無視する。
        """
        Path(path).write_text(self._markdown_text, encoding="utf-8")


def _bgr_ndarray_to_jpeg(bgr_image: np.ndarray) -> bytes:
    """BGR ndarray を JPEG bytes に変換する。

    Args:
        bgr_image: BGR 形式の uint8 ndarray（yomitoku / pypdfium2 ローダーの出力）。

    Returns:
        JPEG 形式のバイト列。
    """
    from PIL import Image as PILImage

    # BGR → RGB に変換して PIL Image を生成する
    rgb_image = bgr_image[:, :, ::-1]
    pil_img = PILImage.fromarray(rgb_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class GeminiAnalyzer:
    """Gemini API を使って画像を OCR するアナライザー。

    AnalyzerProtocol を実装し、既存パイプラインの analyzer として差し込める。
    """

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(
        self, image: np.ndarray
    ) -> tuple[GeminiAnalyzerResult, None, None]:
        """画像を Gemini API に送信して OCR 結果を返す。

        Args:
            image: BGR 形式の uint8 ndarray。

        Returns:
            (GeminiAnalyzerResult, None, None) のタプル（AnalyzerProtocol 互換）。

        Raises:
            RuntimeError: Gemini API 呼び出しが失敗した場合。
        """
        from google.genai import types

        jpeg_bytes = _bgr_ndarray_to_jpeg(image)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    _GEMINI_OCR_PROMPT,
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API 呼び出しに失敗しました: {e}") from e

        return GeminiAnalyzerResult(response.text), None, None


def create_gemini_analyzer(
    *,
    api_key: str,
    model: str = "gemini-3.5-flash",
) -> GeminiAnalyzer:
    """GeminiAnalyzer を生成するファクトリ関数。

    Args:
        api_key: Gemini API キー。
        model: 使用するモデル名（デフォルト: gemini-3.5-flash）。

    Returns:
        GeminiAnalyzer インスタンス。

    Raises:
        ImportError: google-genai が未インストールの場合（Fail-Fast）。
    """
    try:
        from google import genai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "google-genai がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra gemini"
        ) from e

    return GeminiAnalyzer(api_key=api_key, model=model)
