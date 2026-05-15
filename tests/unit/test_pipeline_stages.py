"""仕様: pipeline/stages の各ステージのユニットテスト。

Yomitoku の DocumentAnalyzer は重いモデルを伴うため、
テストでは AnalyzerProtocol を満たす Fake を注入してテストする。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.stages.ocr import (
    analyze_page,
)
from ouj_notebook_converter.pipeline.stages.post_process import build_page_markdown
from ouj_notebook_converter.pipeline.types import PageAnalysis, PageMarkdown

# ---------------------------------------------------------------------------
# Fake アナライザー（テスト用）
# ---------------------------------------------------------------------------

class FakeAnalyzerResult:
    """Yomitoku の DocumentAnalyzer の戻り値 (results) に相当する Fake。"""

    def __init__(self, text: str = "テストテキスト", figures: list[str] | None = None):
        self._text = text
        self._figures = figures or []

    def to_json(self, path: str | Path) -> None:
        """JSON を書き出す。"""
        data = {"text": self._text, "figures": self._figures}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self, path: str | Path, **_kwargs: object) -> None:
        """Markdown を書き出す。"""
        md = f"# テスト見出し\n\n{self._text}\n"
        for fig in self._figures:
            md += f"\n![図]({fig})\n"
        Path(path).write_text(md, encoding="utf-8")

    @property
    def figure_paths(self) -> list[Path]:
        """切り出された figure 画像のパスリスト。"""
        return [Path(f) for f in self._figures]


class FakeAnalyzer:
    """AnalyzerProtocol を満たす Fake。"""

    def __init__(self, result: FakeAnalyzerResult | None = None):
        self._result = result or FakeAnalyzerResult()
        self.call_count = 0

    def __call__(self, image: np.ndarray) -> tuple[FakeAnalyzerResult, object, object]:
        self.call_count += 1
        return self._result, None, None


class TestAnalyzePageStage:
    """analyze_page ステージのテスト。"""

    def test_正常に_PageAnalysis_を返す(self, tmp_path: Path) -> None:
        fake = FakeAnalyzer(FakeAnalyzerResult("放送大学の教科書"))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()

        result = analyze_page(img, cache_page_dir, analyzer=fake)

        assert isinstance(result, PageAnalysis)
        assert result.page_index == -1  # page_index は runner が設定する
        assert result.yomitoku_json_path.exists()
        assert result.markdown_raw_path.exists()

    def test_analysis_json_が書き出される(self, tmp_path: Path) -> None:
        fake = FakeAnalyzer(FakeAnalyzerResult("データの分析"))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()

        result = analyze_page(img, cache_page_dir, analyzer=fake)

        assert result.yomitoku_json_path.exists()
        data = json.loads(result.yomitoku_json_path.read_text())
        assert data["text"] == "データの分析"

    def test_figure_ありの場合_figure_pathsが設定される(self, tmp_path: Path) -> None:
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()
        # figure の実ファイルを作る
        fig_dir = cache_page_dir / "figures"
        fig_dir.mkdir()
        fig_file = fig_dir / "fig_001.png"
        fig_file.write_bytes(b"dummy")

        fake = FakeAnalyzer(FakeAnalyzerResult(figures=[str(fig_file)]))
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = analyze_page(img, cache_page_dir, analyzer=fake)
        assert len(result.figure_paths) == 1

    def test_アナライザーが1回だけ呼ばれる(self, tmp_path: Path) -> None:
        fake = FakeAnalyzer()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()

        analyze_page(img, cache_page_dir, analyzer=fake)
        assert fake.call_count == 1


class TestBuildPageMarkdown:
    """build_page_markdown（post_process ステージ）のテスト。"""

    def test_PageMarkdownを返す(self, tmp_path: Path) -> None:
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("# 見出し\n\n本文テキスト\n", encoding="utf-8")
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis)

        assert isinstance(result, PageMarkdown)
        assert result.page_index == 0
        assert "見出し" in result.markdown

    def test_figure_パスが絶対パスで含まれる(self, tmp_path: Path) -> None:
        fig = tmp_path / "figures" / "fig_001.png"
        fig.parent.mkdir()
        fig.write_bytes(b"dummy")

        raw_md = tmp_path / "raw.md"
        raw_md.write_text(f"本文\n\n![図]({fig})\n", encoding="utf-8")

        analysis = PageAnalysis(
            page_index=1,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[fig],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis)

        assert fig in result.referenced_assets

    def test_raw_md_が存在しない場合はエラー(self, tmp_path: Path) -> None:
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=tmp_path / "存在しない.md",
        )
        with pytest.raises(FileNotFoundError):
            build_page_markdown(analysis)
