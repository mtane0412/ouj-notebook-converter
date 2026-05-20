"""仕様: PageMarkdown リストから EPUB3 ファイルを生成するエクスポーター。

処理内容:
- 全ページの Markdown を結合し、pypandoc 経由で EPUB3 に変換する
- referenced_assets の figure 画像を assets_dir 以下にコピーし、
  pandoc の --resource-path で埋め込みを行う
- pypandoc（pandoc ラッパー）は optional dep (epub extra) のため、
  未インストール時は Fail-Fast で案内メッセージを出す
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from ouj_notebook_converter.pipeline.types import PageMarkdown


def export_epub(
    pages: list[PageMarkdown],
    out_path: Path,
    assets_dir: Path,
    *,
    title: str = "Document",
) -> None:
    """PageMarkdown リストを EPUB3 ファイルとして書き出す。

    Args:
        pages: ページ順に並んだ PageMarkdown のリスト。
        out_path: 出力 .epub ファイルのパス。
        assets_dir: figure アセットをコピーする先のディレクトリ（--resource-path に使用）。
        title: EPUB のタイトルメタデータ。

    Raises:
        ValueError: pages が空の場合。
        ImportError: pypandoc がインストールされていない場合。
    """
    if not pages:
        raise ValueError("pages が空です。変換対象のページが存在しません。")

    try:
        import pypandoc
    except ImportError as e:
        raise ImportError(
            "pypandoc がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra epub\n"
            "  または: pip install 'ouj-notebook-converter[epub]'"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 全ページの figure をコピーし、絶対パス / 相対パス → ファイル名のみ の変換マップを構築する
    # yomitoku は raw.md で <img src="figures/filename.png"> 形式（HTML タグ・相対パス）を使うため、
    # 絶対パス文字列と "figures/filename" の両方をキーとして登録する
    path_map: dict[str, str] = {}
    for page in pages:
        for asset in page.referenced_assets:
            dest = assets_dir / asset.name
            shutil.copy2(asset, dest)
            # 絶対パス → ファイル名のみ
            path_map[str(asset)] = asset.name
            # "figures/filename" → ファイル名のみ（yomitoku HTML img タグの相対パス対応）
            path_map[f"figures/{asset.name}"] = asset.name

    # 全ページの Markdown を結合し、figure パスをファイル名のみに書き換える
    chunks: list[str] = []
    for page in pages:
        text = _rewrite_figure_paths(page.markdown, path_map)
        text = _rewrite_img_src(text, path_map)
        chunks.append(text)
    combined_md = "\n\n".join(chunks)

    # 一時ファイルに Markdown を書き出して pandoc で変換する
    # （pypandoc.convert_text は --resource-path の画像解決が不安定なため convert_file を使用）
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_md = Path(tmpdir) / "input.md"
        tmp_md.write_text(combined_md, encoding="utf-8")

        pypandoc.convert_file(
            str(tmp_md),
            "epub3",
            outputfile=str(out_path),
            extra_args=[
                f"--resource-path={assets_dir}",
                f"--metadata=title:{title}",
            ],
        )


def _rewrite_img_src(markdown: str, path_map: dict[str, str]) -> str:
    """Markdown 内の HTML <img src="..."> タグのパスを path_map に従って書き換える。

    yomitoku は figure を <img src="figures/filename.png"> 形式で出力するため、
    Markdown 画像構文 ![alt](path) とは別に処理する。
    """
    if not path_map:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        before = match.group(1)
        src = match.group(2)
        after = match.group(3)
        new_src = path_map.get(src, src)
        return f'<img {before}src="{new_src}"{after}>'

    return re.sub(r"<img ([^>]*)src=\"([^\"]+)\"([^>]*)>", _replace, markdown)


def _rewrite_figure_paths(markdown: str, path_map: dict[str, str]) -> str:
    """Markdown 内の絶対パス参照をファイル名のみに書き換える。

    path_map のキー（絶対パス文字列）が見つかれば、値（ファイル名文字列）に置換する。
    """
    if not path_map:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        path = match.group(2)
        new_name = path_map.get(path, path)
        return f"![{alt}]({new_name})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown)
