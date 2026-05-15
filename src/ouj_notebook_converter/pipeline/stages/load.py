"""仕様: PDF ファイルをページ画像に変換するロードステージ。

Yomitoku の load_pdf（pypdfium2 バックエンド）を使用する。
遅延レンダリングのため大きな PDF でも OOM しにくい。
yomitoku が未インストールの場合は Fail-Fast で案内する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_pdf_pages(pdf_path: Path, *, dpi: int = 200) -> Any:
    """PDF ファイルをページ画像のイテレータとして読み込む。

    Args:
        pdf_path: 読み込む PDF ファイルのパス。
        dpi: レンダリング解像度（高いほど精度が上がるが処理が重い）。

    Returns:
        PdfPageIterator（total_pages 属性と __iter__ を持つ遅延イテレータ）。

    Raises:
        FileNotFoundError: PDF ファイルが存在しない場合。
        ImportError: yomitoku が未インストールの場合。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF ファイルが見つかりません: {pdf_path}")

    try:
        from yomitoku.data.functions import load_pdf  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "yomitoku がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra ocr"
        ) from e

    return load_pdf(str(pdf_path), dpi=dpi)
