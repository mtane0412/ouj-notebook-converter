"""仕様: パイプラインのページループを管理するオーケストレーター（M1: キャッシュなし）。

各ページについて以下の順に処理を行う:
  1. ローダーからページ画像を取得
  2. analyze_fn でOCRを実行（デフォルトは stages/ocr.py の analyze_page）
  3. math_backend に応じて math_fn または detect_fn で数式を LaTeX に変換
     - "pix2tex"  : math_fn（yomitoku paragraph 経由 + pix2tex 認識）
     - "pix2text" : detect_fn（Pix2Text 検出 + 認識）
     - "none"     : スキップ
  4. post_process でPageMarkdownを構築

M2 でページ単位のキャッシュ判定ロジックをここに追加する予定。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ouj_notebook_converter.pipeline.stages.math_detect import math_detect
from ouj_notebook_converter.pipeline.stages.math_extract import math_extract
from ouj_notebook_converter.pipeline.stages.ocr import analyze_page
from ouj_notebook_converter.pipeline.stages.post_process import build_page_markdown
from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis, PageMarkdown


@runtime_checkable
class PageLoader(Protocol):
    """PDF ページローダーが満たすべきインターフェース。

    Yomitoku の PdfPageIterator に準拠。
    """

    total_pages: int

    def __iter__(self) -> Any:
        """ページ画像（np.ndarray, BGR）のイテレータを返す。"""
        ...


AnalyzeFn = Callable[
    [np.ndarray, Path],  # image, cache_page_dir
    PageAnalysis,
]


@dataclass
class ConvertConfig:
    """1 PDF ファイルの変換パラメータをまとめた設定オブジェクト。"""

    pdf_path: Path
    cache_dir: Path
    page_indices: list[int]  # 0-origin の処理対象ページ番号
    dpi: int
    analyzer: Any  # AnalyzerProtocol を満たすオブジェクト
    reading_order: str = "auto"
    ignore_meta: bool = False
    enable_math: bool = False  # 後方互換のため残す。"pix2tex" のエイリアスとして扱う
    math_engine: Any = field(
        default=None
    )  # MathEngineProtocol / MathDetectorProtocol 互換オブジェクト
    math_backend: Literal["none", "pix2tex", "pix2text"] = "none"


def run_pages(
    config: ConvertConfig,
    *,
    loader: PageLoader,
    analyze_fn: Callable[..., PageAnalysis] | None = None,
    math_fn: Callable[..., MathOverlay] | None = None,
    detect_fn: Callable[..., MathOverlay] | None = None,
) -> list[PageMarkdown]:
    """対象ページを順に OCR 処理し、PageMarkdown のリストを返す。

    Args:
        config: 変換設定。
        loader: PDF ページ画像のイテレータ（yomitoku.data.functions.load_pdf の戻り値相当）。
        analyze_fn: analyze ステージの関数。テストで差し替える用途に使う。
                    省略時は stages/ocr.py の analyze_page を使用する。
        math_fn: math_extract ステージの関数（pix2tex バックエンド用）。
                 省略時は stages/math_extract.py の math_extract を使用する。
        detect_fn: math_detect ステージの関数（pix2text バックエンド用）。
                   省略時は stages/math_detect.py の math_detect を使用する。

    Returns:
        処理順（page_index 昇順）の PageMarkdown リスト。

    Raises:
        RuntimeError: ページ処理中に回復不能なエラーが発生した場合。
    """
    _analyze = analyze_fn or _default_analyze_fn(config)

    # enable_math=True（後方互換）は math_backend="pix2tex" と等価
    effective_backend = config.math_backend
    if effective_backend == "none" and config.enable_math:
        effective_backend = "pix2tex"

    target_set = set(config.page_indices)
    results: list[PageMarkdown] = []

    for raw_index, image in enumerate(loader):
        if raw_index not in target_set:
            continue

        # 1-origin のゼロ埋めディレクトリ名でキャッシュを作る
        page_cache_dir = config.cache_dir / f"page_{raw_index + 1:04d}"
        page_cache_dir.mkdir(parents=True, exist_ok=True)

        analysis = _analyze(image, page_cache_dir, analyzer=config.analyzer)

        # runner が page_index を正しく上書きする
        analysis = PageAnalysis(
            page_index=raw_index,
            yomitoku_json_path=analysis.yomitoku_json_path,
            figure_paths=analysis.figure_paths,
            markdown_raw_path=analysis.markdown_raw_path,
        )

        overlay: MathOverlay | None = None
        if effective_backend == "pix2tex":
            _math = math_fn or _default_math_fn(config)
            overlay = _math(image, analysis, page_cache_dir, engine=config.math_engine)
        elif effective_backend == "pix2text":
            _detect = detect_fn or _default_detect_fn(config)
            overlay = _detect(image, analysis, page_cache_dir, detector=config.math_engine)

        page_md = build_page_markdown(analysis, math_overlay=overlay)
        results.append(page_md)

    return results


def _default_analyze_fn(config: ConvertConfig) -> Callable[..., PageAnalysis]:
    """デフォルトの analyze 関数を生成するファクトリ。"""

    def _fn(
        image: np.ndarray,
        cache_page_dir: Path,
        *,
        analyzer: Any,
    ) -> PageAnalysis:
        return analyze_page(image, cache_page_dir, analyzer=analyzer)

    return _fn


def _default_math_fn(config: ConvertConfig) -> Callable[..., MathOverlay]:
    """デフォルトの math_extract 関数を生成するファクトリ（pix2tex バックエンド用）。"""

    def _fn(
        image: np.ndarray,
        analysis: PageAnalysis,
        cache_page_dir: Path,
        *,
        engine: Any,
    ) -> MathOverlay:
        return math_extract(image, analysis, cache_page_dir, engine=engine)

    return _fn


def _default_detect_fn(config: ConvertConfig) -> Callable[..., MathOverlay]:
    """デフォルトの math_detect 関数を生成するファクトリ（pix2text バックエンド用）。"""

    def _fn(
        image: np.ndarray,
        analysis: PageAnalysis,
        cache_page_dir: Path,
        *,
        detector: Any,
    ) -> MathOverlay:
        return math_detect(image, analysis, cache_page_dir, detector=detector)

    return _fn
