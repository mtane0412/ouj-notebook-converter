"""仕様: pypdfium2 のみを使った PDF ページローダー。

yomitoku 不依存の PDF ローダー。Gemini OCR バックエンドなど
yomitoku をインストールしない環境向けに提供する。
pypdfium2 はコア依存のため常に利用可能。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium


class PypdfiumPageIterator:
    """pypdfium2 を使って PDF ページを BGR ndarray としてイテレートするクラス。

    PageLoader プロトコル（runner.py）に準拠。
    load_pdf_pages_pypdfium2() の戻り値として返される。
    """

    def __init__(self, doc: pdfium.PdfDocument, dpi: int) -> None:
        self._doc = doc
        self._dpi = dpi
        self.total_pages: int = len(doc)

    def __iter__(self) -> Iterator[np.ndarray]:
        """各ページを BGR uint8 ndarray として yield する。

        yomitoku の PdfPageIterator と同じ BGR 形式で返す。
        ビットマップバッファとは独立したコピーを yield する。
        """
        scale = self._dpi / 72.0
        for i in range(len(self._doc)):
            page = self._doc[i]
            bitmap = page.render(scale=scale)
            # to_numpy() はビットマップバッファを共有する場合があるため
            # 明示的にコピーしてからビットマップを解放する
            image: np.ndarray = np.copy(bitmap.to_numpy())
            yield image


def load_pdf_pages_pypdfium2(pdf_path: Path, *, dpi: int = 200) -> PypdfiumPageIterator:
    """pypdfium2 のみを使って PDF をページ画像のイテレータとして読み込む。

    Args:
        pdf_path: 読み込む PDF ファイルのパス。
        dpi: レンダリング解像度（高いほど精度が上がるが処理が重い）。

    Returns:
        PypdfiumPageIterator（total_pages 属性と __iter__ を持つイテレータ）。

    Raises:
        FileNotFoundError: PDF ファイルが存在しない場合。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF ファイルが見つかりません: {pdf_path}")

    doc = pdfium.PdfDocument(str(pdf_path))
    return PypdfiumPageIterator(doc=doc, dpi=dpi)
