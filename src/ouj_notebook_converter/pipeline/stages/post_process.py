"""仕様: PageAnalysis を PageMarkdown に変換する後処理ステージ。

OCR の生 Markdown を読み込み、数式 overlay があれば LaTeX に置換した上で PageMarkdown を返す。
figure パスの相対→絶対変換や書き換えは行わない（exporter が担当）。

数式置換の 2 系統:
  - overlay.originals (display_formula): paragraph テキスト全体を $$LaTeX$$ に置換
  - overlay.inline_paragraphs (inline_formula): paragraph 内の特定 word span を $LaTeX$ に部分置換
"""

from __future__ import annotations

import logging
import re

from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis, PageMarkdown

logger = logging.getLogger(__name__)

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

    display_formula は overlay.originals 経由で paragraph 全体を置換する。
    inline_formula は overlay.inline_paragraphs 経由で paragraph 内の特定 word を置換する。

    Args:
        markdown_text: raw.md の内容。
        overlay: math_extract / math_detect ステージの出力。

    Returns:
        数式 paragraph を LaTeX に置換した Markdown 文字列。
        needle が見つからない場合は警告を出してスキップし、元のテキストをそのまま返す。
    """
    text = markdown_text

    # display_formula: paragraph 全体を LaTeX 表記で置換（math_extract バックエンド用）
    for img_path, original_contents in overlay.originals.items():
        latex = overlay.items.get(img_path, "")
        role = overlay.roles.get(img_path, "")
        # 空文字（NoOp 結果）はスキップ（暗黙フォールバックではなく明示的 no-op）
        if not latex:
            continue

        escaped_needle = _escape_like_yomitoku(original_contents)
        if escaped_needle not in text:
            logger.warning(
                "raw.md 中で数式 paragraph テキストが見つかりません（スキップ）: %r",
                escaped_needle[:40],
            )
            continue

        if role == "inline_formula":
            replacement = f"${latex}$"
        elif role == "display_formula":
            replacement = f"\n$${latex}$$\n"
        else:
            replacement = latex

        text = text.replace(escaped_needle, replacement, 1)

    # inline_formula: paragraph 内の特定 word span を $LaTeX$ に部分置換（math_detect バックエンド用）
    # para_idx 昇順で処理することで、同一 needle が複数段落に存在する場合の誤置換を防ぐ
    for _para_idx, repl in sorted(overlay.inline_paragraphs.items(), key=lambda kv: kv[0]):
        # word.content を各々エスケープして連結 → raw.md 上の needle
        needle_parts = [_escape_like_yomitoku(w) for w in repl.word_contents]
        needle = "".join(needle_parts)
        if needle not in text:
            logger.warning(
                "raw.md 中でインライン数式 paragraph テキストが見つかりません（スキップ）: %r",
                needle[:40],
            )
            continue

        # latex_spans を start 昇順で処理し、word parts を順に置換
        replacement_parts: list[str] = []
        i = 0
        for start, end, latex in sorted(repl.latex_spans, key=lambda s: s[0]):
            replacement_parts.extend(needle_parts[i:start])
            replacement_parts.append(f"${latex}$")
            i = end
        replacement_parts.extend(needle_parts[i:])
        replacement = "".join(replacement_parts)

        text = text.replace(needle, replacement, 1)

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
    """
    if not analysis.markdown_raw_path.exists():
        raise FileNotFoundError(
            f"raw Markdown ファイルが見つかりません: {analysis.markdown_raw_path}"
        )

    raw_text = analysis.markdown_raw_path.read_text(encoding="utf-8")

    if math_overlay is not None:
        markdown_text = _apply_math_overlay(raw_text, math_overlay)
    else:
        markdown_text = raw_text

    # referenced_assets は実際に存在する figure パスのみを含める
    existing_assets = [p for p in analysis.figure_paths if p.exists()]

    return PageMarkdown(
        page_index=analysis.page_index,
        markdown=markdown_text,
        referenced_assets=existing_assets,
        yomitoku_json_path=analysis.yomitoku_json_path,
        # 数式オーバーレイ適用時のみ raw を保持する（章検出が overlay の影響を受けないよう）
        raw_markdown=raw_text if math_overlay is not None else None,
    )
