"""仕様: PageMarkdown リストから最終的な Markdown ファイルを生成するエクスポーター。

処理内容:
- figure の絶対パス参照を assets_dir 以下の相対パスに書き換える
- referenced_assets のファイルを assets_dir 以下にコピーする
- combine=True の場合は 1 ファイルに結合、False の場合はページごとのファイルに分割する
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from ouj_notebook_converter.pipeline.types import PageMarkdown


def export_markdown(
    pages: list[PageMarkdown],
    out_path: Path,
    assets_dir: Path,
    *,
    combine: bool = True,
) -> None:
    """PageMarkdown リストを Markdown ファイルとして書き出す。

    Args:
        pages: ページ順に並んだ PageMarkdown のリスト。
        out_path: combine=True なら出力 .md ファイルのパス、
                  combine=False なら page_NNNN.md を格納するディレクトリのパス。
        assets_dir: figure アセットをコピーする先のディレクトリ。
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
            page_text = _process_page(page, assets_dir)
            chunks.append(page_text)
        out_path.write_text("\n\n".join(chunks), encoding="utf-8")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        for page in pages:
            page_text = _process_page(page, assets_dir)
            page_file = out_path / f"page_{page.page_index + 1:04d}.md"
            page_file.write_text(page_text, encoding="utf-8")


def _process_page(page: PageMarkdown, assets_dir: Path) -> str:
    """1 ページ分の Markdown を処理する。

    referenced_assets をコピーし、markdown 内の絶対パス参照を相対パスに書き換える。
    """
    # figure ファイルをコピーし、パス変換マップを作る
    path_map: dict[str, str] = {}
    for asset in page.referenced_assets:
        dest = _copy_asset(asset, assets_dir)
        # markdown 内の絶対パス文字列 → 相対パス文字列にマッピング
        path_map[str(asset)] = str(dest)

    markdown = _rewrite_figure_paths(page.markdown, path_map)

    return markdown


def _copy_asset(asset: Path, assets_dir: Path) -> Path:
    """figure 画像を assets_dir にコピーし、コピー先のパスを返す。

    ページインデックスを保持するために assets_dir 直下にそのままコピーする。
    同名ファイルが複数ページに存在する場合、後のページが上書きされる可能性があるため、
    呼び出し側でページ別サブディレクトリを assets_dir に含めることを推奨する。
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / asset.name
    shutil.copy2(asset, dest)
    return dest


def _rewrite_figure_paths(markdown: str, path_map: dict[str, str]) -> str:
    """Markdown 内の絶対パス参照を path_map に従って書き換える。

    正規表現で Markdown の画像構文 ![...](path) および リンク [...]( path) を対象にする。
    path_map のキー（絶対パス文字列）が見つかれば、値（相対パス文字列）に置換する。
    """
    if not path_map:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        path = match.group(2)
        new_path = path_map.get(path, path)
        # 相対パスにするためにファイル名のみ使う（assets_dir 相対は呼び出し元が管理）
        new_path_obj = Path(new_path)
        return f"![{alt}]({new_path_obj.name})"

    # Markdown 画像構文 ![alt](path) にマッチ
    result = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown)
    return result
