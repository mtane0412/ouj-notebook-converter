"""仕様: PageAnalysis を PageMarkdown に変換する後処理ステージ。

OCR の生 Markdown を読み込み、referenced_assets に figure の絶対パスを設定する。
figure パスの相対→絶対変換や書き換えは行わない（exporter が担当）。
"""
from __future__ import annotations

from ouj_notebook_converter.pipeline.types import PageAnalysis, PageMarkdown


def build_page_markdown(analysis: PageAnalysis) -> PageMarkdown:
    """PageAnalysis から PageMarkdown を生成する純粋関数。

    Args:
        analysis: analyze ステージの出力。

    Returns:
        PageMarkdown（referenced_assets には figure の絶対パスを格納）。

    Raises:
        FileNotFoundError: markdown_raw_path が存在しない場合。
    """
    if not analysis.markdown_raw_path.exists():
        raise FileNotFoundError(
            f"raw Markdown ファイルが見つかりません: {analysis.markdown_raw_path}"
        )

    markdown_text = analysis.markdown_raw_path.read_text(encoding="utf-8")

    # referenced_assets は実際に存在する figure パスのみを含める
    existing_assets = [p for p in analysis.figure_paths if p.exists()]

    return PageMarkdown(
        page_index=analysis.page_index,
        markdown=markdown_text,
        referenced_assets=existing_assets,
    )
