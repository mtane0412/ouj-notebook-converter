"""仕様: ounc CLI のエントリポイント。Typer ベースのサブコマンド集約モジュール。

M1 では convert サブコマンドのみ実装する。
M2 以降で resume / inspect を追加する予定。

Note: シラバス連携（detect_via_syllabus）は内部実装済みだが、OUJ シラバス URL の
解析方式が未確定のため --course-code オプションは現バージョンでは公開していない。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from ouj_notebook_converter.exporters.markdown import (
    export_markdown,
    export_markdown_by_chapters,
)
from ouj_notebook_converter.pipeline.runner import ConvertConfig, run_pages
from ouj_notebook_converter.pipeline.stages.chapter_detect import (
    ChapterDetectionError,
    detect_chapters,
)
from ouj_notebook_converter.pipeline.stages.load import load_pdf_pages
from ouj_notebook_converter.pipeline.stages.ocr import create_analyzer
from ouj_notebook_converter.plugins.math.server_manager import (
    Pix2TextServerManager,
    ServerStartupError,
)
from ouj_notebook_converter.utils.pages import parse_page_range

app = typer.Typer(
    name="ounc",
    help="放送大学テキスト PDF を AI フレンドリーな形式に変換する CLI ツール。",
    no_args_is_help=True,
)

_VALID_DEVICES = {"mps", "cpu", "cuda"}
_DEFAULT_PIX2TEXT_VENV = Path.home() / ".venvs" / "pix2text"


class OutputFormat(str, Enum):
    """サポートする出力形式。"""

    md = "md"
    epub = "epub"
    pdf = "pdf"
    txt = "txt"


class ReadingOrder(str, Enum):
    """Yomitoku の読み順推定モード。"""

    auto = "auto"
    left2right = "left2right"
    right2left = "right2left"
    top2bottom = "top2bottom"


class SplitMode(str, Enum):
    """出力ファイルの分割モード。"""

    none = "none"  # デフォルト（--combine / --no-combine に従う）
    chapters = "chapters"  # 章ごとに分割


class MathBackend(str, Enum):
    """数式変換バックエンドの選択肢。"""

    none = "none"  # 数式処理しない（デフォルト）
    pix2text = "pix2text"  # Pix2Text API で検出・認識


@app.command()
def convert(
    input_pdf: Annotated[Path, typer.Argument(help="変換する PDF ファイルのパス")],
    outdir: Annotated[
        Path | None, typer.Option("--outdir", "-o", help="出力先ディレクトリ（必須）")
    ] = None,
    format: Annotated[
        list[OutputFormat] | None,
        typer.Option("-f", "--format", help="出力形式（複数指定可）"),
    ] = None,
    device: Annotated[
        str, typer.Option("-d", "--device", help="推論デバイス: mps / cpu / cuda")
    ] = "mps",
    dpi: Annotated[int, typer.Option(help="PDF レンダリング DPI")] = 200,
    pages: Annotated[
        str | None, typer.Option("--pages", help="処理するページ範囲 例: 1,3-5,10")
    ] = None,
    cache_dir: Annotated[
        Path | None, typer.Option("--cache-dir", help="キャッシュディレクトリ")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="キャッシュを無効化")] = False,
    combine: Annotated[
        bool, typer.Option("--combine/--no-combine", help="全ページを 1 ファイルに結合")
    ] = True,
    reading_order: Annotated[
        ReadingOrder,
        typer.Option("--reading-order", help="読み順推定モード"),
    ] = ReadingOrder.auto,
    ignore_meta: Annotated[
        bool, typer.Option("--ignore-meta/--no-ignore-meta", help="ヘッダ/フッタを除外")
    ] = True,
    split: Annotated[
        SplitMode,
        typer.Option("--split", help="出力分割モード: none / chapters"),
    ] = SplitMode.none,
    math_backend: Annotated[
        MathBackend,
        typer.Option(
            "--math-backend",
            help="数式変換バックエンド: none (デフォルト), pix2text (Pix2Text 検出+認識)",
        ),
    ] = MathBackend.none,
    pix2text_url: Annotated[
        str,
        typer.Option(
            "--pix2text-url",
            help="Pix2Text 自前ラッパー URL (math-backend=pix2text 時)",
        ),
    ] = "http://localhost:8503",
    pix2text_venv: Annotated[
        Path,
        typer.Option(
            "--pix2text-venv",
            envvar="OUC_PIX2TEXT_VENV",
            help="pix2text 用 venv のパス（デフォルト: ~/.venvs/pix2text）",
        ),
    ] = _DEFAULT_PIX2TEXT_VENV,
    math_auto_start: Annotated[
        bool,
        typer.Option(
            "--math-auto-start/--no-math-auto-start",
            help="--math-backend pix2text 時にサーバーを自動起動する（デフォルト: 有効）",
        ),
    ] = True,
    verbose: Annotated[bool, typer.Option("-v/-q", "--verbose/--quiet")] = False,
) -> None:
    """PDF ファイルを指定した形式に変換する。"""
    # --- バリデーション ---
    if outdir is None:
        typer.echo("エラー: --outdir (-o) オプションは必須です。", err=True)
        raise typer.Exit(code=1)

    if not input_pdf.exists():
        typer.echo(f"エラー: PDF ファイルが見つかりません: {input_pdf}", err=True)
        raise typer.Exit(code=1)

    if device not in _VALID_DEVICES:
        typer.echo(
            f"エラー: 無効なデバイス '{device}'。指定可能: {', '.join(sorted(_VALID_DEVICES))}",
            err=True,
        )
        raise typer.Exit(code=1)

    effective_cache_dir = cache_dir or (outdir / ".cache")

    # math_backend に応じたエンジンを構築する
    math_engine: Any = None
    if math_backend == MathBackend.pix2text:
        if math_auto_start:
            manager = Pix2TextServerManager(
                url=pix2text_url,
                venv_path=pix2text_venv,
                server_script=_resolve_pix2text_server_script(),
            )
            if not manager.is_server_alive():
                typer.echo(
                    "Pix2Text サーバーを起動中... (モデルロードに 10〜30 秒かかります)", err=True
                )
                try:
                    manager.ensure_running()
                except ServerStartupError as e:
                    typer.echo(f"エラー: {e}", err=True)
                    raise typer.Exit(code=1) from e
                typer.echo("Pix2Text サーバーが起動しました。", err=True)

        from ouj_notebook_converter.plugins.math.pix2text_http import Pix2TextHttpDetector

        math_engine = Pix2TextHttpDetector(base_url=pix2text_url)

    analyzer = create_analyzer(
        device=device,
        reading_order=reading_order.value,
        ignore_meta=ignore_meta,
    )

    loader = load_pdf_pages(input_pdf, dpi=dpi)
    page_indices_1based = parse_page_range(pages, total=loader.total_pages)
    page_indices = [p - 1 for p in page_indices_1based]  # 0-origin に変換

    book_name = input_pdf.stem
    book_cache_dir = effective_cache_dir / f"{book_name}.{_short_hash(input_pdf)}"

    config = ConvertConfig(
        pdf_path=input_pdf,
        cache_dir=book_cache_dir,
        page_indices=page_indices,
        dpi=dpi,
        analyzer=analyzer,
        reading_order=reading_order.value,
        ignore_meta=ignore_meta,
        math_engine=math_engine,
        math_backend=math_backend.value,
    )

    if verbose:
        typer.echo(f"OCR 開始: {input_pdf.name} ({len(page_indices)} ページ)")

    page_markdowns = run_pages(config, loader=loader)

    assets_dir = outdir / f"{book_name}_assets"

    effective_format = format or [OutputFormat.md]

    if OutputFormat.md in effective_format:
        if split == SplitMode.chapters:
            try:
                chapters = detect_chapters(input_pdf, page_markdowns)
            except ChapterDetectionError as e:
                typer.echo(f"エラー: {e}", err=True)
                raise typer.Exit(code=2) from e

            chapter_dir = outdir / book_name
            written = export_markdown_by_chapters(page_markdowns, chapters, chapter_dir, assets_dir)
            if verbose:
                typer.echo(f"章分割 Markdown 出力: {chapter_dir} ({len(written)} ファイル)")
        else:
            out_path = outdir / f"{book_name}.md" if combine else outdir / book_name
            export_markdown(page_markdowns, out_path, assets_dir, combine=combine)
            if verbose:
                typer.echo(f"Markdown 出力: {out_path}")

    typer.echo("変換が完了しました。")


def _resolve_pix2text_server_script() -> Path:
    """リポジトリの scripts/pix2text_server.py のパスを返す。

    cli.py が src/ouj_notebook_converter/cli.py に位置することを前提に、
    cli.py が src/ouj_notebook_converter/cli.py に位置することを前提に、
    3 階層上のリポジトリルートから scripts/ を解決する。
    """
    return Path(__file__).parent.parent.parent / "scripts" / "pix2text_server.py"


def _short_hash(path: Path) -> str:
    """ファイルの SHA-256 短縮ハッシュを返す（キャッシュディレクトリ名に使用）。"""
    from ouj_notebook_converter.utils.hashing import sha256_file

    try:
        return sha256_file(path, short=True)
    except FileNotFoundError:
        return "unknown"
