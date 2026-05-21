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


class TestRunPagesWithCache:
    """run_pages のページキャッシュ機能テスト。"""

    def _make_cache_files(self, page_cache_dir: Path, content: str) -> None:
        """テスト用のキャッシュファイルを作成する。"""
        page_cache_dir.mkdir(parents=True, exist_ok=True)
        (page_cache_dir / "analysis.json").write_text("{}", encoding="utf-8")
        (page_cache_dir / "raw.md").write_text(content, encoding="utf-8")

    def test_キャッシュが存在する場合はanalyze_fnが呼ばれない(self, tmp_path: Path) -> None:
        # 前提: page_0001 にキャッシュファイルが存在する
        cache_dir = tmp_path / ".cache"
        self._make_cache_files(cache_dir / "page_0001", "# キャッシュ済みページ\n\nキャッシュ内容\n")

        fake_loader = _make_fake_loader(1)
        fake_analyze = MagicMock()

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=cache_dir,
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            no_cache=False,
        )

        # 検証: キャッシュ存在時は analyze_fn が呼ばれない
        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        fake_analyze.assert_not_called()

    def test_キャッシュが存在する場合はキャッシュのmarkdownが使われる(self, tmp_path: Path) -> None:
        # 前提: キャッシュに特定内容が書かれている
        cache_dir = tmp_path / ".cache"
        cached_text = "# キャッシュ済みページ\n\nキャッシュ内容\n"
        self._make_cache_files(cache_dir / "page_0001", cached_text)

        fake_loader = _make_fake_loader(1)

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=cache_dir,
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            no_cache=False,
        )

        # 検証: キャッシュ内容が PageMarkdown.markdown に反映される
        results = run_pages(config, loader=fake_loader)

        assert len(results) == 1
        assert results[0].markdown == cached_text

    def test_no_cache_trueのときキャッシュがあってもanalyze_fnが呼ばれる(
        self, tmp_path: Path
    ) -> None:
        # 前提: キャッシュファイルが存在するが --no-cache フラグが立っている
        cache_dir = tmp_path / ".cache"
        self._make_cache_files(cache_dir / "page_0001", "# 古いキャッシュ\n\n古い内容\n")

        fake_loader = _make_fake_loader(1)
        fake_analyze = _make_fake_analyze(tmp_path)

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=cache_dir,
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            no_cache=True,
        )

        # 検証: --no-cache 指定時は analyze_fn が呼ばれる
        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        fake_analyze.assert_called_once()

    def test_キャッシュが存在しない場合はanalyze_fnが呼ばれる(self, tmp_path: Path) -> None:
        # 前提: キャッシュディレクトリ内にファイルが存在しない
        cache_dir = tmp_path / ".cache"

        fake_loader = _make_fake_loader(1)
        fake_analyze = _make_fake_analyze(tmp_path)

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=cache_dir,
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            no_cache=False,
        )

        # 検証: キャッシュ未存在時は analyze_fn が呼ばれる
        run_pages(config, loader=fake_loader, analyze_fn=fake_analyze)

        fake_analyze.assert_called_once()


class TestRunPagesWithMath:
    """run_pages の math_extract ステージ組み込みテスト。"""

    def test_math_backend_noneならdetect_fnは呼ばれない(self, tmp_path: Path) -> None:
        """math_backend="none"（デフォルト）の場合、detect_fn は呼ばれない。"""
        fake_loader = _make_fake_loader(1)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_detect = MagicMock(return_value=MathOverlay())

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            math_backend="none",
        )

        run_pages(
            config,
            loader=fake_loader,
            analyze_fn=fake_analyze,
            detect_fn=fake_detect,
        )

        fake_detect.assert_not_called()

    def test_math_backend_pix2textならdetect_fnのみ呼ばれる(self, tmp_path: Path) -> None:
        """math_backend="pix2text" のとき detect_fn が 2 ページ分呼ばれる。"""
        fake_loader = _make_fake_loader(2)
        fake_analyze = _make_fake_analyze(tmp_path)
        fake_detect = MagicMock(return_value=MathOverlay())

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0, 1],
            dpi=200,
            analyzer=MagicMock(),
            math_backend="pix2text",
            math_engine=MagicMock(),
        )

        run_pages(
            config,
            loader=fake_loader,
            analyze_fn=fake_analyze,
            detect_fn=fake_detect,
        )

        assert fake_detect.call_count == 2

    def test_detect_fnの結果がbuild_page_markdownにoverlayとして渡る(self, tmp_path: Path) -> None:
        """detect_fn が返した overlay を通じて LaTeX が最終 Markdown に反映される。"""
        fake_loader = _make_fake_loader(1)

        def _fake_analyze(
            image: np.ndarray, page_cache_dir: Path, *, analyzer: object
        ) -> PageAnalysis:
            json_path = page_cache_dir / "analysis.json"
            raw_md_path = page_cache_dir / "raw.md"
            json_path.write_text("{}", encoding="utf-8")
            raw_md_path.write_text("本文テキスト\n\n数式テキスト\n", encoding="utf-8")
            return PageAnalysis(
                page_index=0,
                yomitoku_json_path=json_path,
                figure_paths=[],
                markdown_raw_path=raw_md_path,
            )

        crop_png = tmp_path / "0000.png"

        def _fake_detect(
            image: np.ndarray,
            analysis: PageAnalysis,
            cache_page_dir: Path,
            *,
            detector: object,
        ) -> MathOverlay:
            return MathOverlay(
                items={crop_png: r"\sum_{k=1}^{n}"},
                roles={crop_png: "display_formula"},
                originals={crop_png: "数式テキスト"},
            )

        config = ConvertConfig(
            pdf_path=Path("dummy.pdf"),
            cache_dir=tmp_path / ".cache",
            page_indices=[0],
            dpi=200,
            analyzer=MagicMock(),
            math_backend="pix2text",
            math_engine=MagicMock(),
        )

        results = run_pages(
            config, loader=fake_loader, analyze_fn=_fake_analyze, detect_fn=_fake_detect
        )

        assert len(results) == 1
        assert r"\sum_{k=1}^{n}" in results[0].markdown

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
