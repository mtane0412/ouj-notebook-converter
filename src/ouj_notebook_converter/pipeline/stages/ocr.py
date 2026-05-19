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
        """OCR 結果を Markdown に書き出す。img キーワード引数で元画像を渡す（yomitoku 0.13.0 必須）。"""
        ...


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """DocumentAnalyzer が満たすべきインターフェース（callable オブジェクト）。

    __call__(image) → (results, ocr_vis, layout_vis) のシグネチャを期待する。
    ocr_vis / layout_vis は可視化画像で本パイプラインでは使用しない。
    """

    def __call__(self, image: np.ndarray) -> tuple[AnalyzerResult, Any, Any]:
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
    results.to_markdown(raw_md_path, img=image, ignore_line_break=True)

    # to_markdown が cache_page_dir/figures/ に PNG を保存するため glob で収集する
    figures_dir = cache_page_dir / "figures"
    figure_paths = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []

    return PageAnalysis(
        page_index=-1,  # runner が page_index を置き換える
        yomitoku_json_path=json_path,
        figure_paths=figure_paths,
        markdown_raw_path=raw_md_path,
    )


def create_analyzer(
    *,
    device: str = "mps",
    reading_order: str = "auto",
    ignore_meta: bool = True,
) -> Any:
    """本番用の Yomitoku DocumentAnalyzer を生成する（yomitoku 0.13.0 API 対応）。

    yomitoku が未インストールの場合は ImportError を Fail-Fast で送出する。

    Args:
        device: 推論デバイス（"mps" / "cpu" / "cuda"）。
        reading_order: 読み順推定モード（"auto" / "right2left" / "top2bottom"）。
        ignore_meta: True の場合はヘッダ/フッタを除外する。

    Raises:
        ImportError: yomitoku が未インストールの場合。
    """
    try:
        from yomitoku import DocumentAnalyzer
    except ImportError as e:
        raise ImportError(
            "yomitoku がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra ocr\n"
            "  または: pip install 'ouj-notebook-converter[ocr]'"
        ) from e

    return DocumentAnalyzer(
        visualize=False,
        device=device,
        reading_order=reading_order,
        ignore_meta=ignore_meta,
    )
