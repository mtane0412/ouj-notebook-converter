"""仕様: パイプラインのページループを管理するオーケストレーター。

各ページについて以下の順に処理を行う:
  1. ローダーからページ画像を取得
  2. キャッシュ判定: raw.md + analysis.json が存在する場合はOCRをスキップ
     （ConvertConfig.no_cache=True の場合は常にOCRを再実行）
  3. analyze_fn でOCRを実行（デフォルトは stages/ocr.py の analyze_page）
  4. math_backend に応じて detect_fn で数式を LaTeX に変換
     - "pix2text" : detect_fn（Pix2Text 検出 + 認識）
     - "none"     : スキップ
  5. post_process でPageMarkdownを構築
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ouj_notebook_converter.pipeline.stages.math_detect import math_detect
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
    math_engine: Any = field(default=None)  # MathDetectorProtocol 互換オブジェクト
    math_backend: Literal["none", "pix2text"] = "none"
    no_cache: bool = False  # True の場合はキャッシュを無視して OCR を再実行する


def run_pages(
    config: ConvertConfig,
    *,
    loader: PageLoader,
    analyze_fn: Callable[..., PageAnalysis] | None = None,
    detect_fn: Callable[..., MathOverlay] | None = None,
) -> list[PageMarkdown]:
    """対象ページを順に OCR 処理し、PageMarkdown のリストを返す。

    Args:
        config: 変換設定。
        loader: PDF ページ画像のイテレータ（yomitoku.data.functions.load_pdf の戻り値相当）。
        analyze_fn: analyze ステージの関数。テストで差し替える用途に使う。
                    省略時は stages/ocr.py の analyze_page を使用する。
        detect_fn: math_detect ステージの関数（pix2text バックエンド用）。
                   省略時は stages/math_detect.py の math_detect を使用する。

    Returns:
        処理順（page_index 昇順）の PageMarkdown リスト。

    Raises:
        RuntimeError: ページ処理中に回復不能なエラーが発生した場合。
    """
    _analyze = analyze_fn or _default_analyze_fn(config)

    target_set = set(config.page_indices)
    results: list[PageMarkdown] = []

    for raw_index, image in enumerate(loader):
        if raw_index not in target_set:
            continue

        # 1-origin のゼロ埋めディレクトリ名でキャッシュを作る
        page_cache_dir = config.cache_dir / f"page_{raw_index + 1:04d}"
        page_cache_dir.mkdir(parents=True, exist_ok=True)

        # キャッシュ判定: raw.md と analysis.json が揃っていれば OCR をスキップする
        json_path = page_cache_dir / "analysis.json"
        raw_md_path = page_cache_dir / "raw.md"
        cache_hit = (
            not config.no_cache
            and json_path.exists()
            and raw_md_path.exists()
        )

        if cache_hit:
            figures_dir = page_cache_dir / "figures"
            figure_paths = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
            analysis = PageAnalysis(
                page_index=raw_index,
                yomitoku_json_path=json_path,
                figure_paths=figure_paths,
                markdown_raw_path=raw_md_path,
            )
        else:
            analysis = _analyze(image, page_cache_dir, analyzer=config.analyzer)
            # runner が page_index を正しく上書きする
            analysis = PageAnalysis(
                page_index=raw_index,
                yomitoku_json_path=analysis.yomitoku_json_path,
                figure_paths=analysis.figure_paths,
                markdown_raw_path=analysis.markdown_raw_path,
            )

        overlay: MathOverlay | None = None
        if config.math_backend == "pix2text":
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



def _default_detect_fn(config: ConvertConfig) -> Callable[..., MathOverlay]:
    """デフォルトの math_detect 関数を生成するファクトリ（pix2text バックエンド用）。

    config.math_engine は MathDetectorProtocol と MathRecognizerProtocol の両方を満たす
    Pix2TextHttpDetector インスタンスを想定しており、detector と recognizer の両引数に渡す。
    """

    def _fn(
        image: np.ndarray,
        analysis: PageAnalysis,
        cache_page_dir: Path,
        *,
        detector: Any,
    ) -> MathOverlay:
        return math_detect(
            image, analysis, cache_page_dir, detector=detector, recognizer=detector
        )

    return _fn
