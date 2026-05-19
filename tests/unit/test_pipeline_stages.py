"""仕様: pipeline/stages の各ステージのユニットテスト。

Yomitoku の DocumentAnalyzer は重いモデルを伴うため、
テストでは AnalyzerProtocol を満たす Fake を注入してテストする。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.stages.ocr import (
    analyze_page,
    create_analyzer,
)
from ouj_notebook_converter.pipeline.stages.post_process import build_page_markdown
from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis, PageMarkdown

# ---------------------------------------------------------------------------
# Fake アナライザー（テスト用）
# ---------------------------------------------------------------------------


class FakeAnalyzerResult:
    """Yomitoku の DocumentAnalyzer の戻り値 (results) に相当する Fake。"""

    def __init__(self, text: str = "テストテキスト", figures: list[str] | None = None):
        self._text = text
        self._figures = figures or []
        self.to_markdown_kwargs: dict[str, object] = {}

    def to_json(self, path: str | Path) -> None:
        """JSON を書き出す。"""
        data = {"text": self._text, "figures": self._figures}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self, path: str | Path, **kwargs: object) -> None:
        """Markdown を書き出す。yomitoku 0.13.0 の動作を模倣して figures/ に PNG を作成する。"""
        self.to_markdown_kwargs = kwargs
        figures_dir = Path(path).parent / "figures"
        md = f"# テスト見出し\n\n{self._text}\n"
        for i, _fig in enumerate(self._figures):
            # yomitoku が raw_figure_{i}.png として保存する動作を模倣する
            figures_dir.mkdir(exist_ok=True)
            fig_path = figures_dir / f"raw_figure_{i}.png"
            fig_path.write_bytes(b"dummy_png")
            md += f'\n<img src="figures/raw_figure_{i}.png">\n'
        Path(path).write_text(md, encoding="utf-8")


class FakeAnalyzer:
    """AnalyzerProtocol を満たす Fake。"""

    def __init__(self, result: FakeAnalyzerResult | None = None):
        self._result = result or FakeAnalyzerResult()
        self.call_count = 0

    def __call__(self, image: np.ndarray) -> tuple[FakeAnalyzerResult, object, object]:
        self.call_count += 1
        return self._result, None, None


def _mock_yomitoku(mock_cls: MagicMock) -> dict[str, MagicMock]:
    """yomitoku モジュールを sys.modules に差し込むためのヘルパー。

    create_analyzer 内の `from yomitoku import DocumentAnalyzer` を
    モックで置き換えるために使用する。
    """
    mock_module = MagicMock()
    mock_module.DocumentAnalyzer = mock_cls
    return {"yomitoku": mock_module}


class TestCreateAnalyzer:
    """create_analyzer のテスト（DocumentAnalyzer をモックして呼び出し引数を検証）。"""

    def test_デフォルト引数でDocumentAnalyzerを生成できる(self) -> None:
        mock_cls = MagicMock()
        with patch.dict("sys.modules", _mock_yomitoku(mock_cls)):
            result = create_analyzer()

        mock_cls.assert_called_once()
        assert result is mock_cls.return_value

    def test_ignore_metaのデフォルトはTrue(self) -> None:
        """ヘッダー/フッターを除外するために ignore_meta=True がデフォルト。"""
        mock_cls = MagicMock()
        with patch.dict("sys.modules", _mock_yomitoku(mock_cls)):
            create_analyzer()

        _, kwargs = mock_cls.call_args
        assert kwargs["ignore_meta"] is True

    def test_reading_orderがauto以外のとき直接渡される(self) -> None:
        mock_cls = MagicMock()
        with patch.dict("sys.modules", _mock_yomitoku(mock_cls)):
            create_analyzer(reading_order="right2left")

        _, kwargs = mock_cls.call_args
        assert kwargs["reading_order"] == "right2left"

    def test_ignore_metaがTrueのとき直接渡される(self) -> None:
        mock_cls = MagicMock()
        with patch.dict("sys.modules", _mock_yomitoku(mock_cls)):
            create_analyzer(ignore_meta=True)

        _, kwargs = mock_cls.call_args
        assert kwargs["ignore_meta"] is True

    def test_liteやignore_line_breakの引数を受け付けない(self) -> None:
        """yomitoku 0.13.0 で削除された引数が create_analyzer に存在しないことを確認。"""
        import inspect

        sig = inspect.signature(create_analyzer)
        assert "lite" not in sig.parameters
        assert "ignore_line_break" not in sig.parameters


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
        """to_markdown が figures/ に PNG を作成した場合、result.figure_paths に収集される。"""
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()
        # FakeAnalyzerResult は to_markdown 時に figures/ に PNG を作成する
        fake = FakeAnalyzer(FakeAnalyzerResult(figures=["仮図1"]))
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

    def test_to_markdownにimgが渡される(self, tmp_path: Path) -> None:
        """yomitoku 0.13.0 では to_markdown に img（元画像）が必要。"""
        result_fake = FakeAnalyzerResult("テスト本文")
        fake = FakeAnalyzer(result_fake)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cache_page_dir = tmp_path / "page_0001"
        cache_page_dir.mkdir()

        analyze_page(img, cache_page_dir, analyzer=fake)

        assert "img" in result_fake.to_markdown_kwargs
        assert result_fake.to_markdown_kwargs["img"] is img
        assert result_fake.to_markdown_kwargs.get("ignore_line_break") is True


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

    def test_math_overlay_Noneなら従来通りraw_mdをそのまま返す(self, tmp_path: Path) -> None:
        """math_overlay を省略した場合（None）、raw.md の内容がそのまま markdown になる。"""
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("本文テキスト\n", encoding="utf-8")
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis)

        assert result.markdown == "本文テキスト\n"

    def test_inline_formulaがdollar囲みで置換される(self, tmp_path: Path) -> None:
        """inline_formula role の paragraph テキストが $LaTeX$ に置換される。"""
        # yomitoku は escape_markdown_special_chars を通してから raw.md に書く
        # 日本語のみの場合は変換なし。末尾に \n を追加。
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("前段落テキスト\n\nインライン数式\n\n後段落テキスト\n", encoding="utf-8")

        crop_png = tmp_path / "0000.png"
        overlay = MathOverlay(
            items={crop_png: r"\alpha + \beta"},
            roles={crop_png: "inline_formula"},
            originals={crop_png: "インライン数式"},
        )
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis, math_overlay=overlay)

        assert r"$\alpha + \beta$" in result.markdown
        # 元のテキストは置換済みで残っていない
        assert "インライン数式" not in result.markdown

    def test_display_formulaが二重dollarで前後改行付き置換される(self, tmp_path: Path) -> None:
        """display_formula role の paragraph テキストが $$LaTeX$$ に置換される。"""
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("前段落\n\n表示数式全体\n\n後段落\n", encoding="utf-8")

        crop_png = tmp_path / "0001.png"
        overlay = MathOverlay(
            items={crop_png: r"\int_0^\infty e^{-x} dx"},
            roles={crop_png: "display_formula"},
            originals={crop_png: "表示数式全体"},
        )
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis, math_overlay=overlay)

        assert r"$$\int_0^\infty e^{-x} dx$$" in result.markdown

    def test_该当テキストがraw_mdになければRuntimeError(self, tmp_path: Path) -> None:
        """raw.md に数式 paragraph テキストが見つからない場合は RuntimeError（Fail-Fast）。"""
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("全く無関係なテキスト\n", encoding="utf-8")

        crop_png = tmp_path / "0000.png"
        overlay = MathOverlay(
            items={crop_png: r"\gamma"},
            roles={crop_png: "inline_formula"},
            originals={crop_png: "raw.mdにないテキスト"},
        )
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        with pytest.raises(
            RuntimeError, match=r"raw\.md 中で数式 paragraph テキストが見つかりません"
        ):
            build_page_markdown(analysis, math_overlay=overlay)

    def test_空のLaTeX_NoOp結果_はスキップされる(self, tmp_path: Path) -> None:
        """engine が空文字を返した場合（NoOp 結果）は置換をスキップして原文を維持する。"""
        raw_md = tmp_path / "raw.md"
        raw_md.write_text("前文\n\nスキップされる数式\n\n後文\n", encoding="utf-8")

        crop_png = tmp_path / "0000.png"
        overlay = MathOverlay(
            items={crop_png: ""},  # 空文字 = NoOp
            roles={crop_png: "display_formula"},
            originals={crop_png: "スキップされる数式"},
        )
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=tmp_path / "analysis.json",
            figure_paths=[],
            markdown_raw_path=raw_md,
        )

        result = build_page_markdown(analysis, math_overlay=overlay)

        # 空文字はスキップされるので元テキストが残る
        assert "スキップされる数式" in result.markdown
