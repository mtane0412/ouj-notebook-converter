"""仕様: Yomitoku DocumentAnalyzer を呼び出す OCR ステージ。

- AnalyzerProtocol を定義することで、テスト時に Fake を注入できる設計にする。
- Yomitoku 自体は任意依存（ocr extra）のため、import は本関数内で行う。
  yomitoku が未インストールの場合は Fail-Fast で案内メッセージを出す。
- 本番では create_analyzer() で実際の DocumentAnalyzer を生成する。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ouj_notebook_converter.pipeline.types import PageAnalysis

if TYPE_CHECKING:
    pass  # 循環 import 防止用


class AnalyzerResult(Protocol):
    """Yomitoku の DocumentAnalyzer の戻り値 (results) が満たすべきインターフェース。"""

    def to_json(self, path: str | Path) -> None:
        """OCR 結果を JSON に書き出す。"""
        ...

    def to_markdown(self, path: str | Path, **kwargs: Any) -> None:
        """OCR 結果を Markdown に書き出す。"""
        ...

    @property
    def figure_paths(self) -> list[Path]:
        """切り出し済み figure 画像のパスリスト。"""
        ...


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """DocumentAnalyzer が満たすべきインターフェース（callable オブジェクト）。

    __call__(image) → (results, ocr_vis, layout_vis) のシグネチャを期待する。
    ocr_vis / layout_vis は可視化画像で本パイプラインでは使用しない。
    """

    def __call__(
        self, image: np.ndarray
    ) -> tuple[AnalyzerResult, Any, Any]:
        """画像を受け取り OCR 結果を返す。"""
        ...


def analyze_page(
    image: np.ndarray,
    cache_page_dir: Path,
    *,
    analyzer: AnalyzerProtocol,
) -> PageAnalysis:
    """1 ページ分の OCR を実行し、中間ファイルをキャッシュディレクトリに書き出す。

    Args:
        image: BGR 形式の NumPy 配列（Yomitoku の load_pdf が返す形式）。
        cache_page_dir: ページ単位のキャッシュディレクトリ（事前に存在している前提）。
        analyzer: AnalyzerProtocol を満たす OCR エンジン（本番は DocumentAnalyzer）。

    Returns:
        PageAnalysis（page_index は runner が後から設定するため -1 を返す）。

    Raises:
        RuntimeError: analyzer の呼び出しに失敗した場合。
    """
    results, _, _ = analyzer(image)

    json_path = cache_page_dir / "analysis.json"
    results.to_json(json_path)

    raw_md_path = cache_page_dir / "raw.md"
    results.to_markdown(raw_md_path)

    # figure ファイルのパスを収集（存在するものだけ）
    figure_paths = [p for p in results.figure_paths if p.exists()]

    return PageAnalysis(
        page_index=-1,  # runner が page_index を置き換える
        yomitoku_json_path=json_path,
        figure_paths=figure_paths,
        markdown_raw_path=raw_md_path,
    )


def create_analyzer(
    *,
    device: str = "mps",
    lite: bool = False,
    reading_order: str = "auto",
    ignore_line_break: bool = False,
    ignore_meta: bool = False,
) -> Any:
    """本番用の Yomitoku DocumentAnalyzer を生成する。

    yomitoku が未インストールの場合は ImportError を Fail-Fast で送出する。

    Args:
        device: 推論デバイス（"mps" / "cpu" / "cuda"）。
        lite: True の場合は軽量モデルを使用する（CPU 実行向け）。
        reading_order: 読み順推定モード。
        ignore_line_break: True の場合は段落内の改行を無視する。
        ignore_meta: True の場合はヘッダ/フッタを除外する。

    Raises:
        ImportError: yomitoku が未インストールの場合。
    """
    try:
        from yomitoku import DocumentAnalyzer  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "yomitoku がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra ocr\n"
            "  または: pip install 'ouj-notebook-converter[ocr]'"
        ) from e

    configs: dict[str, Any] = {}
    if reading_order != "auto":
        configs["reading_order"] = reading_order
    if ignore_line_break:
        configs["ignore_line_break"] = True
    if ignore_meta:
        configs["ignore_meta"] = True

    return DocumentAnalyzer(
        visualize=False,
        device=device,
        lite=lite,
        configs=configs if configs else None,
    )
