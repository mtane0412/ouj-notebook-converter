"""仕様: pipeline.runner モジュールのユニットテスト。

runner はステージのオーケストレーターであり、
各ステージを差し替え可能にして以下を検証する:
- 指定されたページ番号分だけ analyze が呼ばれる
- analyze の結果が post_process を経て PageMarkdown に変換される
- ステージ関数を介してページが順に処理される
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from ouj_notebook_converter.pipeline.runner import ConvertConfig, run_pages
from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis, PageMarkdown

# ---------------------------------------------------------------------------
# テスト用 Fake
# ---------------------------------------------------------------------------


def _make_fake_loader(num_pages: int) -> MagicMock:
    """指定枚数の黒画像を返す Fake ローダーを作る。"""
    loader = MagicMock()
    loader.total_pages = num_pages
    loader.__iter__ = MagicMock(
        return_value=iter([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(num_pages)])
    )
    return loader


def _make_fake_analyze(cache_dir: Path) -> MagicMock:
    """analyze_page の Fake（キャッシュディレクトリにファイルを作る）。"""
    call_count = {"n": 0}

    def _fake(image: np.ndarray, page_cache_dir: Path, *, analyzer: object) -> PageAnalysis:
        n = call_count["n"]
        call_count["n"] += 1
        json_path = page_cache_dir / "analysis.json"
        raw_md_path = page_cache_dir / "raw.md"
        json_path.write_text(json.dumps({"text": f"ページ{n + 1}"}), encoding="utf-8")
        raw_md_path.write_text(f"# ページ{n + 1}\n\nテスト内容\n", encoding="utf-8")
        return PageAnalysis(
            page_index=n,
            yomitoku_json_path=json_path,
            figure_paths=[],
            markdown_raw_path=raw_md_path,
        )

    mock = MagicMock(side_effect=_fake)
    mock.call_count_tracker = call_count
    return mock


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


class TestRunPages:
    """run_pages 関数のテスト。"""

    def test_全ページ数分のPageMarkdownを返す(self, tmp_path: Path) -> None:
        fake_loader = _make_fake_loader(3)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_analyzer = MagicMock()

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 1, 2],
            dpi=200,
            analyzer=fake_analyzer,
        )

        results = run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        assert len(results) == 3
        assert all(isinstance(r, PageMarkdown) for r in results)

    def test_指定ページのみ処理される(self, tmp_path: Path) -> None:
        fake_loader = _make_fake_loader(5)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_analyzer = MagicMock()

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 2],  # ページ0と2だけ
            dpi=200,
            analyzer=fake_analyzer,
        )

        results = run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        assert len(results) == 2

    def test_ページが順番に処理される(self, tmp_path: Path) -> None:
        fake_loader = _make_fake_loader(3)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_analyzer = MagicMock()

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 1, 2],
            dpi=200,
            analyzer=fake_analyzer,
        )

        results = run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        assert results[0].page_index == 0
        assert results[1].page_index == 1
        assert results[2].page_index == 2


class TestRunPagesWithMath:
    """run_pages の math_extract ステージ組み込みテスト。"""

    def test_enable_math_Falseならmath_fnは呼ばれない(self, tmp_path: Path) -> None:
        """enable_math=False（デフォルト）の場合、math_fn は呼ばれない。"""
        fake_loader = _make_fake_loader(2)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_math = MagicMock(return_value=MathOverlay())

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 1],
            dpi=200,
            analyzer=MagicMock(),
            enable_math=False,
        )

        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze, math_fn=fake_math)

        fake_math.assert_not_called()

    def test_enable_math_Trueなら各ページでmath_fnが呼ばれる(self, tmp_path: Path) -> None:
        """enable_math=True のとき、ページ数分だけ math_fn が呼ばれる。"""
        fake_loader = _make_fake_loader(2)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_math = MagicMock(return_value=MathOverlay())

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 1],
            dpi=200,
            analyzer=MagicMock(),
            enable_math=True,
            math_engine=MagicMock(),
        )

        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze, math_fn=fake_math)

        assert fake_math.call_count == 2

    def test_overlayがbuild_page_markdownに渡って最終markdownにLaTeXが入る(
        self, tmp_path: Path
    ) -> None:
        """math_fn が返した overlay を通じて LaTeX が最終 Markdown に反映される。"""
        fake_loader = _make_fake_loader(1)

        def _fake_analyze(
            image: np.ndarray, page_cache_dir: Path, *, analyzer: object
        ) -> PageAnalysis:
            json_path = page_cache_dir / "analysis.json"
            raw_md_path = page_cache_dir / "raw.md"
            json_path.write_text("{}", encoding="utf-8")
            # raw.md に数式テキストを含める
            raw_md_path.write_text("本文テキスト\n\n数式部分\n", encoding="utf-8")
            return PageAnalysis(
                page_index=0,
                yomitoku_json_path=json_path,
                figure_paths=[],
                markdown_raw_path=raw_md_path,
            )

        crop_png = tmp_path / "0000.png"

        def _fake_math(
            image: np.ndarray, analysis: PageAnalysis, cache_page_dir: Path, *, engine: object
        ) -> MathOverlay:
            return MathOverlay(
                items={crop_png: r"\Gamma(z)"},
                roles={crop_png: "display_formula"},
                originals={crop_png: "数式部分"},
            )

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            enable_math=True,
            math_engine=MagicMock(),
        )

        results = run_pages(
            config, loader=fake_loader, analyze_fn=_fake_analyze, math_fn=_fake_math
        )

        assert len(results) == 1
        assert r"\Gamma(z)" in results[0].markdown

    def test_ページキャッシュディレクトリが作成される(self, tmp_path: Path) -> None:
        fake_loader = _make_fake_loader(2)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_analyzer = MagicMock()

        cache_dir = tmp_path / ".cache"
        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=cache_dir,
            page_indices=[0, 1],
            dpi=200,
            analyzer=fake_analyzer,
        )

        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        assert (cache_dir / "page_0001").is_dir()
        assert (cache_dir / "page_0002").is_dir()
