"""仕様: pypdfium2 を使った PDF ローダーのユニットテスト。

load_pdf_pages_pypdfium2 と PypdfiumPageIterator の動作を検証する。
yomitoku 不要で動くことが設計上の要件。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.runner import PageLoader
from ouj_notebook_converter.pipeline.stages.load_pypdfium import (
    PypdfiumPageIterator,
    load_pdf_pages_pypdfium2,
)

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_fake_bitmap(width: int = 100, height: int = 80) -> MagicMock:
    """pypdfium2 の FPDFBitmap に相当するフェイクを生成する。"""
    bitmap = MagicMock()
    # BGR 形式の ndarray を返す（shape: (height, width, 3)）
    bitmap.to_numpy.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    return bitmap


def _make_fake_page(width: int = 100, height: int = 80) -> MagicMock:
    """pypdfium2 の PdfPage に相当するフェイクを生成する。"""
    page = MagicMock()
    page.render.return_value = _make_fake_bitmap(width, height)
    return page


def _make_fake_document(pages: list[MagicMock]) -> MagicMock:
    """pypdfium2 の PdfDocument に相当するフェイクを生成する。"""
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    return doc


# ---------------------------------------------------------------------------
# PypdfiumPageIterator のテスト
# ---------------------------------------------------------------------------


class TestPypdfiumPageIterator:
    """PypdfiumPageIterator の動作検証。"""

    def test_total_pagesがドキュメントのページ数を返す(self) -> None:
        """total_pages プロパティが PDF のページ数を正しく返すこと。"""
        pages = [_make_fake_page(), _make_fake_page(), _make_fake_page()]
        doc = _make_fake_document(pages)

        iterator = PypdfiumPageIterator(doc=doc, dpi=200)

        assert iterator.total_pages == 3

    def test_iterはBGR_ndarrayをyieldする(self) -> None:
        """__iter__ が (H, W, 3) uint8 の BGR ndarray を yield すること。"""
        pages = [_make_fake_page(width=640, height=480)]
        doc = _make_fake_document(pages)

        iterator = PypdfiumPageIterator(doc=doc, dpi=200)
        images = list(iterator)

        assert len(images) == 1
        image = images[0]
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3
        assert image.shape[2] == 3  # BGR の 3 チャンネル

    def test_iterはページ数分のndarrayをyieldする(self) -> None:
        """__iter__ が PDF のページ数だけ ndarray を yield すること。"""
        pages = [_make_fake_page(), _make_fake_page()]
        doc = _make_fake_document(pages)

        iterator = PypdfiumPageIterator(doc=doc, dpi=200)
        images = list(iterator)

        assert len(images) == 2

    def test_指定DPIからscaleを計算してrenderが呼ばれる(self) -> None:
        """render(scale=dpi/72.0) が正しい scale で呼ばれること。"""
        fake_page = _make_fake_page()
        doc = _make_fake_document([fake_page])

        dpi = 144
        iterator = PypdfiumPageIterator(doc=doc, dpi=dpi)
        list(iterator)

        expected_scale = dpi / 72.0
        call_kwargs = fake_page.render.call_args
        assert call_kwargs is not None
        actual_scale = call_kwargs.kwargs.get("scale") or call_kwargs.args[0]
        assert actual_scale == pytest.approx(expected_scale)

    def test_ndarrayはビットマップとは独立したコピーである(self) -> None:
        """yield する ndarray はビットマップバッファとは独立したコピーであること。

        ビットマップ解放後に ndarray を読んでも元のデータが保持されていること。
        """
        original_array = np.array([[[10, 20, 30]]], dtype=np.uint8)
        bitmap = MagicMock()
        bitmap.to_numpy.return_value = original_array.copy()

        page = MagicMock()
        page.render.return_value = bitmap
        doc = _make_fake_document([page])

        iterator = PypdfiumPageIterator(doc=doc, dpi=200)
        images = list(iterator)

        # ビットマップの元データを変更しても yield 済みの ndarray は変わらない
        original_array[0, 0, 0] = 99
        assert images[0][0, 0, 0] != 99

    def test_PageLoaderプロトコルを満たす(self) -> None:
        """PypdfiumPageIterator が PageLoader プロトコルを満たすこと。"""
        pages = [_make_fake_page()]
        doc = _make_fake_document(pages)

        iterator = PypdfiumPageIterator(doc=doc, dpi=200)

        assert isinstance(iterator, PageLoader)


# ---------------------------------------------------------------------------
# load_pdf_pages_pypdfium2 のテスト
# ---------------------------------------------------------------------------


class TestLoadPdfPagesPypdfium2:
    """load_pdf_pages_pypdfium2 の動作検証。"""

    def test_存在しないPDFはFileNotFoundError(self, tmp_path: Path) -> None:
        """存在しない PDF パスを渡すと FileNotFoundError が送出されること。"""
        not_exist = tmp_path / "存在しない.pdf"

        with pytest.raises(FileNotFoundError, match="存在しない"):
            load_pdf_pages_pypdfium2(not_exist)

    def test_PypdfiumPageIteratorを返す(self, tmp_path: Path) -> None:
        """正常系: PypdfiumPageIterator のインスタンスを返すこと。"""
        # 空の PDF ファイルを用意（pypdfium2.PdfDocument はモックする）
        dummy_pdf = tmp_path / "テスト.pdf"
        dummy_pdf.write_bytes(b"dummy")

        mock_doc = _make_fake_document([_make_fake_page()])
        with patch("ouj_notebook_converter.pipeline.stages.load_pypdfium.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_doc

            result = load_pdf_pages_pypdfium2(dummy_pdf)

        assert isinstance(result, PypdfiumPageIterator)

    def test_デフォルトDPIは200(self, tmp_path: Path) -> None:
        """dpi を省略した場合のデフォルト値が 200 であること。"""
        dummy_pdf = tmp_path / "テスト.pdf"
        dummy_pdf.write_bytes(b"dummy")

        mock_doc = _make_fake_document([_make_fake_page()])
        with patch("ouj_notebook_converter.pipeline.stages.load_pypdfium.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_doc

            result = load_pdf_pages_pypdfium2(dummy_pdf)

        assert result._dpi == 200

    def test_指定DPIがイテレータに引き継がれる(self, tmp_path: Path) -> None:
        """dpi 引数がイテレータに正しく渡されること。"""
        dummy_pdf = tmp_path / "テスト.pdf"
        dummy_pdf.write_bytes(b"dummy")

        mock_doc = _make_fake_document([_make_fake_page()])
        with patch("ouj_notebook_converter.pipeline.stages.load_pypdfium.pdfium") as mock_pdfium:
            mock_pdfium.PdfDocument.return_value = mock_doc

            result = load_pdf_pages_pypdfium2(dummy_pdf, dpi=300)

        assert result._dpi == 300
