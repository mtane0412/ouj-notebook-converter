"""仕様: 原本 PDF のページ描画と、候補の位置での画像の切り出し。

- PageRenderer: pypdfium2 で PDF のページを 200 DPI（yomitoku の座標と同じ）で描画する
- crop_region: bbox に余白を付けて切り出す。低すぎる範囲は MIN_CROP_HEIGHT まで広げる
- fit_long_side / encode_jpeg: LLM に渡すための縮小と JPEG 化

Pillow は gemini extra に含まれる。
"""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ouj_notebook_converter.judge.models import BBox

DEFAULT_DPI = 200
# 切り出し画像の最小の高さ（px）。1 行だけの bbox でも前後の文脈が入るようにする
MIN_CROP_HEIGHT = 200
DEFAULT_PAD_X = 40
DEFAULT_PAD_Y = 80
JPEG_QUALITY = 90


class PageRenderer:
    """PDF のページを PIL 画像として描画する（直近に描画した画像をキャッシュする）。"""

    def __init__(self, pdf_path: Path, *, dpi: int = DEFAULT_DPI) -> None:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF ファイルが見つかりません: {pdf_path}")
        self._doc = pdfium.PdfDocument(str(pdf_path))
        self._scale = dpi / 72.0
        self._cache: dict[int, Image.Image] = {}

    @property
    def page_count(self) -> int:
        """PDF の総ページ数。"""
        return len(self._doc)

    def render(self, page: int) -> Image.Image:
        """page（1 始まり）を RGB 画像として描画する。

        Raises:
            ValueError: ページ番号が範囲外の場合。
        """
        if not 1 <= page <= len(self._doc):
            raise ValueError(f"ページ {page} は範囲外です（1〜{len(self._doc)} ページ）")
        if page not in self._cache:
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[page] = (
                self._doc[page - 1].render(scale=self._scale).to_pil().convert("RGB")
            )
        return self._cache[page]


def crop_region(
    image: Image.Image,
    bbox: BBox,
    *,
    pad_x: int = DEFAULT_PAD_X,
    pad_y: int = DEFAULT_PAD_Y,
    full_width: bool = False,
) -> Image.Image:
    """bbox に余白を付けて切り出す。画像の外にははみ出さない。

    full_width が True なら、横方向は bbox によらず画像の全幅にする（単語だけの bbox で
    行が途中で切れ、LLM が行の一部だけを見て誤った修正を出すのを防ぐ）。
    """
    x0, y0, x1, y1 = bbox
    left, right = max(0, x0 - pad_x), min(image.width, x1 + pad_x)
    if full_width:
        left, right = 0, image.width
    top, bottom = max(0, y0 - pad_y), min(image.height, y1 + pad_y)
    shortage = MIN_CROP_HEIGHT - (bottom - top)
    if shortage > 0:
        # 不足分を上下に等分して広げ、画像の端に当たった分は反対側へ回す
        top = max(0, top - shortage // 2)
        bottom = min(image.height, top + MIN_CROP_HEIGHT)
        top = max(0, bottom - MIN_CROP_HEIGHT)
    return image.crop((left, top, right, bottom))


def fit_long_side(image: Image.Image, max_side: int) -> Image.Image:
    """長辺が max_side を超える場合だけ、縦横比を保って縮小する。"""
    longest = max(image.size)
    if longest <= max_side:
        return image
    ratio = max_side / longest
    return image.resize(
        (round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS
    )


def encode_jpeg(image: Image.Image) -> bytes:
    """画像を JPEG のバイト列にする。"""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()
