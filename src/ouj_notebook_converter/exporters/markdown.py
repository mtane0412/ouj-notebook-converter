"""仕様: PageMarkdown リストから最終的な Markdown ファイルを生成するエクスポーター。

処理内容:
- figure の絶対パス参照を assets_dir 以下の相対パスに書き換える
- referenced_assets のファイルを assets_dir 以下にコピーする
- combine=True の場合は 1 ファイルに結合、False の場合はページごとのファイルに分割する
- export_markdown_by_chapters: ChapterSpec リストに従って章ごとにファイルを分割する
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ouj_notebook_converter.pipeline.types import ChapterSpec, PageMarkdown

# ファイル名として使用できない文字（Windows/macOS 共通で禁止される文字と制御文字）
_FILENAME_FORBIDDEN = re.compile(r'[/\\:*?"<>|\x00-\x1f\x7f]')
# 全角スペース
_FULLWIDTH_SPACE = re.compile(r"　")
# 半角スペース
_HALFWIDTH_SPACE = re.compile(r" ")
# 連続アンダースコア
_CONSECUTIVE_UNDERSCORES = re.compile(r"_+")

_MAX_SLUG_LEN = 80


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


def export_markdown_by_chapters(
    pages: list[PageMarkdown],
    chapters: list[ChapterSpec],
    out_dir: Path,
    assets_dir: Path,
) -> list[Path]:
    """章ごとに 1 ファイルずつ Markdown を生成する。

    各 ChapterSpec の start_page_index〜end_page_index 範囲のページを結合して
    出力ディレクトリ（out_dir）に書き出す。

    Args:
        pages: OCR 済み全ページの PageMarkdown リスト（page_index 順）。
        chapters: 章境界情報のリスト（order 順）。
        out_dir: 章ごとの .md ファイルを配置するディレクトリ。
        assets_dir: figure アセットをコピーする先のディレクトリ。

    Returns:
        書き出したファイルパスのリスト（order 昇順）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # page_index → PageMarkdown の高速参照マップを作成する
    page_map: dict[int, PageMarkdown] = {p.page_index: p for p in pages}

    written: list[Path] = []
    for chapter in sorted(chapters, key=lambda c: c.order):
        chapter_pages = [
            page_map[i]
            for i in range(chapter.start_page_index, chapter.end_page_index + 1)
            if i in page_map
        ]
        chunks: list[str] = []
        for page in chapter_pages:
            chunks.append(_process_page(page, assets_dir))

        file_path = out_dir / _chapter_filename(chapter)
        file_path.write_text("\n\n".join(chunks), encoding="utf-8")
        written.append(file_path)

    return written


def _chapter_filename(chapter: ChapterSpec) -> str:
    """ChapterSpec からファイル名を生成する。

    形式: {order:02d}_{kind}_{slugified_title}.md
    """
    slug = _slugify(chapter.title)
    return f"{chapter.order:02d}_{chapter.kind.value}_{slug}.md"


def _slugify(title: str) -> str:
    """タイトル文字列をファイル名として安全なスラグに変換する。

    処理順:
    1. 全角スペース・半角スペースを _ に置換
    2. ファイル名禁止文字を _ に置換
    3. 連続する _ を 1 つに圧縮
    4. 先頭・末尾の _ を除去
    5. 80 文字を超える場合は切り詰め
    """
    result = _FULLWIDTH_SPACE.sub("_", title)
    result = _HALFWIDTH_SPACE.sub("_", result)
    result = _FILENAME_FORBIDDEN.sub("_", result)
    result = _CONSECUTIVE_UNDERSCORES.sub("_", result)
    result = result.strip("_")
    return result[:_MAX_SLUG_LEN]
