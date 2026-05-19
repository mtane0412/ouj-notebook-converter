"""仕様: PageAnalysis を PageMarkdown に変換する後処理ステージ。

OCR の生 Markdown を読み込み、数式 overlay があれば LaTeX に置換した上で PageMarkdown を返す。
figure パスの相対→絶対変換や書き換えは行わない（exporter が担当）。
"""

from __future__ import annotations

import re

from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis, PageMarkdown

# yomitoku の escape_markdown_special_chars と同じ正規表現（export_markdown.py:8 と同期）
_SPECIAL_CHARS = re.compile(r"([`*{}[\]()#+!~|-])")


def _escape_like_yomitoku(text: str) -> str:
    """yomitoku の escape_markdown_special_chars と同じエスケープを適用する。

    ignore_line_break=True で raw.md が生成されているため、改行も除去する。
    """
    text = text.replace("\n", "")
    return _SPECIAL_CHARS.sub(r"\\\1", text)


def _apply_math_overlay(markdown_text: str, overlay: MathOverlay) -> str:
    """raw.md 中の数式 paragraph テキストを LaTeX 表記で置換する。

    Args:
        markdown_text: raw.md の内容。
        overlay: math_extract ステージの出力。

    Returns:
        数式 paragraph を LaTeX に置換した Markdown 文字列。

    Raises:
        RuntimeError: raw.md に数式 paragraph テキストが見つからない場合（Fail-Fast）。
    """
    text = markdown_text
    for img_path, original_contents in overlay.originals.items():
        latex = overlay.items.get(img_path, "")
        role = overlay.roles.get(img_path, "")
        # 空文字（NoOp 結果）はスキップ（暗黙フォールバックではなく明示的 no-op）
        if not latex:
            continue

        escaped_needle = _escape_like_yomitoku(original_contents)
        if escaped_needle not in text:
            raise RuntimeError(
                f"raw.md 中で数式 paragraph テキストが見つかりません: {escaped_needle[:40]!r}"
            )

        if role == "inline_formula":
            replacement = f"${latex}$"
        elif role == "display_formula":
            replacement = f"\n$${latex}$$\n"
        else:
            replacement = latex

        text = text.replace(escaped_needle, replacement, 1)

    return text


def build_page_markdown(
    analysis: PageAnalysis,
    *,
    math_overlay: MathOverlay | None = None,
) -> PageMarkdown:
    """PageAnalysis から PageMarkdown を生成する純粋関数。

    Args:
        analysis: analyze ステージの出力。
        math_overlay: math_extract ステージの出力。指定すると数式を LaTeX に置換する。

    Returns:
        PageMarkdown（referenced_assets には figure の絶対パスを格納）。

    Raises:
        FileNotFoundError: markdown_raw_path が存在しない場合。
        RuntimeError: math_overlay 指定時に数式テキストが raw.md に見つからない場合。
    """
    if not analysis.markdown_raw_path.exists():
        raise FileNotFoundError(
            f"raw Markdown ファイルが見つかりません: {analysis.markdown_raw_path}"
        )

    markdown_text = analysis.markdown_raw_path.read_text(encoding="utf-8")

    if math_overlay is not None:
        markdown_text = _apply_math_overlay(markdown_text, math_overlay)

    # referenced_assets は実際に存在する figure パスのみを含める
    existing_assets = [p for p in analysis.figure_paths if p.exists()]

    return PageMarkdown(
        page_index=analysis.page_index,
        markdown=markdown_text,
        referenced_assets=existing_assets,
    )
