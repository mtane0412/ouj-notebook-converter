"""仕様: PageMarkdown リストから最終的なプレーンテキストファイルを生成するエクスポーター。

処理内容:
- Markdown 記法（見出し・太字・斜体・コード・水平線）をプレーンテキストに変換する
- 画像参照 ![alt](path) は [図: alt] に変換する（TXT では画像表示不可）
- アセットファイルのコピーは行わない
- combine=True の場合は 1 ファイルに結合、False の場合はページごとのファイルに分割する
"""

from __future__ import annotations

import re
from pathlib import Path

from ouj_notebook_converter.pipeline.types import PageMarkdown

# 見出し（行頭の # を除去）
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# 太字（** または __ で囲まれた部分）
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# 斜体（* または _ で囲まれた部分。太字より後に処理する）
_ITALIC = re.compile(r"\*(.+?)\*|_(.+?)_")
# インラインコード
_INLINE_CODE = re.compile(r"`(.+?)`")
# 画像参照 ![alt](path)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
# リンク [text](url)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# 水平線（行全体が --- / *** / ___ のみ）
_HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)


def export_plaintext(
    pages: list[PageMarkdown],
    out_path: Path,
    assets_dir: Path,  # API 一貫性のため受け取るが TXT では使用しない
    *,
    combine: bool = True,
) -> None:
    """PageMarkdown リストをプレーンテキストファイルとして書き出す。

    Args:
        pages: ページ順に並んだ PageMarkdown のリスト。
        out_path: combine=True なら出力 .txt ファイルのパス、
                  combine=False なら page_NNNN.txt を格納するディレクトリのパス。
        assets_dir: 使用しない（API 一貫性のためのパラメータ）。
        combine: True の場合は全ページを 1 ファイルに結合する。

    Raises:
        ValueError: pages が空の場合。
    """
    if not pages:
        raise ValueError("pages が空です。変換対象のページが存在しません。")

    if combine:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[str] = []
        for page in pages:
            chunks.append(_markdown_to_plaintext(page.markdown))
        out_path.write_text("\n\n".join(chunks), encoding="utf-8")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        for page in pages:
            page_text = _markdown_to_plaintext(page.markdown)
            page_file = out_path / f"page_{page.page_index + 1:04d}.txt"
            page_file.write_text(page_text, encoding="utf-8")


def _markdown_to_plaintext(markdown: str) -> str:
    """Markdown 記法をプレーンテキストに変換する。

    変換ルール:
    - 見出し記号（#）を除去
    - 太字・斜体マーカー（** __ * _）を除去
    - インラインコードのバッククォートを除去
    - 画像参照 ![alt](path) を [図: alt] に変換
    - リンク [text](url) をテキストのみに変換
    - 水平線（--- / *** / ___）を空行に変換
    """
    result = _HEADING.sub("", markdown)
    result = _BOLD.sub(lambda m: m.group(1) or m.group(2), result)
    result = _ITALIC.sub(lambda m: m.group(1) or m.group(2), result)
    result = _INLINE_CODE.sub(r"\1", result)
    result = _IMAGE.sub(lambda m: f"[図: {m.group(1)}]", result)
    result = _LINK.sub(r"\1", result)
    result = _HORIZONTAL_RULE.sub("", result)
    return result
