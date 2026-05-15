"""仕様: pipeline.types モジュールのデータクラスのユニットテスト。"""
from pathlib import Path

import pytest

from ouj_notebook_converter.pipeline.types import (
    MathOverlay,
    PageAnalysis,
    PageJob,
    PageMarkdown,
)


class TestPageJob:
    """PageJob frozen dataclass のテスト。"""

    def test_正常に作成できる(self) -> None:
        job = PageJob(
            pdf_path=Path("samples/データの分析と知識発見.pdf"),
            page_index=0,
            dpi=200,
            cache_key="abc123",
        )
        assert job.page_index == 0
        assert job.dpi == 200

    def test_frozen_で変更不可(self) -> None:
        job = PageJob(
            pdf_path=Path("test.pdf"),
            page_index=0,
            dpi=200,
            cache_key="key",
        )
        with pytest.raises((AttributeError, TypeError)):
            job.page_index = 1  # type: ignore[misc]


class TestPageAnalysis:
    """PageAnalysis frozen dataclass のテスト。"""

    def test_正常に作成できる(self, tmp_path: Path) -> None:
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[tmp_path / "fig_001.png"],
            markdown_raw_path=tmp_path / "raw.md",
        )
        assert analysis.page_index == 0
        assert len(analysis.figure_paths) == 1

    def test_figure_paths_が空でも作成できる(self, tmp_path: Path) -> None:
        analysis = PageAnalysis(
            page_index=1,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=tmp_path / "raw.md",
        )
        assert analysis.figure_paths == []


class TestMathOverlay:
    """MathOverlay frozen dataclass のテスト。"""

    def test_正常に作成できる(self, tmp_path: Path) -> None:
        fig = tmp_path / "fig.png"
        overlay = MathOverlay(items={fig: r"\int_0^1 x^2\,dx"})
        assert overlay.items[fig] == r"\int_0^1 x^2\,dx"

    def test_空の辞書でも作成できる(self) -> None:
        overlay = MathOverlay(items={})
        assert overlay.items == {}


class TestPageMarkdown:
    """PageMarkdown frozen dataclass のテスト。"""

    def test_正常に作成できる(self, tmp_path: Path) -> None:
        pm = PageMarkdown(
            page_index=0,
            markdown="# 第1章\n本文テキスト",
            referenced_assets=[tmp_path / "fig.png"],
        )
        assert pm.page_index == 0
        assert "第1章" in pm.markdown

    def test_assets_が空でも作成できる(self) -> None:
        pm = PageMarkdown(
            page_index=0,
            markdown="テキストのみのページ",
            referenced_assets=[],
        )
        assert pm.referenced_assets == []
