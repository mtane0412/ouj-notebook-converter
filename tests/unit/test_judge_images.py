"""仕様: judge.images のユニットテスト。

PDF のページ描画（200 DPI）、候補の位置での切り出し、JPEG 化をテストする。
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from ouj_notebook_converter.judge.images import (
    MIN_CROP_HEIGHT,
    PageRenderer,
    crop_region,
    encode_jpeg,
    fit_long_side,
)


def _make_pdf(path: Path, pages: int = 2) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=(148 * 72 / 25.4, 210 * 72 / 25.4))
    for _ in range(pages):
        c.drawString(50, 400, "test")
        c.showPage()
    c.save()


def test_PageRenderer_は200dpiで描画する(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf)
    renderer = PageRenderer(pdf, dpi=200)
    image = renderer.render(1)
    # A5（148 x 210 mm）の 200 DPI
    assert abs(image.width - 1165) <= 2
    assert abs(image.height - 1654) <= 2
    assert renderer.page_count == 2


def test_PageRenderer_は範囲外のページでエラーにする(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf)
    with pytest.raises(ValueError, match="ページ"):
        PageRenderer(pdf).render(3)


def test_PageRenderer_はPDFが無ければエラーにする(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PageRenderer(tmp_path / "none.pdf")


def test_crop_region_は余白を付け画像の外にはみ出さない() -> None:
    image = Image.new("RGB", (1000, 1500), "white")
    cropped = crop_region(image, (10, 400, 900, 440), pad_x=40, pad_y=60)
    assert cropped.width == 940  # 左は 0 で打ち切り、右は 900 + 40
    assert cropped.height >= MIN_CROP_HEIGHT


def test_crop_region_は_full_width_なら横幅いっぱいを切り出す() -> None:
    """単語の bbox だけを切り出すと行が途中で切れるため、行全体が見えるよう横幅は全幅にする。"""
    image = Image.new("RGB", (1000, 1500), "white")
    cropped = crop_region(image, (400, 400, 500, 440), pad_x=40, pad_y=60, full_width=True)
    assert cropped.width == 1000


def test_crop_region_は低い範囲を最小の高さまで広げる() -> None:
    image = Image.new("RGB", (1000, 1500), "white")
    cropped = crop_region(image, (100, 700, 500, 720), pad_x=0, pad_y=0)
    assert cropped.height == MIN_CROP_HEIGHT


def test_fit_long_side_は長辺を超える画像だけ縮小する() -> None:
    assert fit_long_side(Image.new("RGB", (2000, 1000)), 1000).size == (1000, 500)
    assert fit_long_side(Image.new("RGB", (800, 600)), 1000).size == (800, 600)


def test_encode_jpeg_はJPEGのバイト列を返す() -> None:
    data = encode_jpeg(Image.new("RGB", (10, 10), "white"))
    assert Image.open(io.BytesIO(data)).format == "JPEG"
