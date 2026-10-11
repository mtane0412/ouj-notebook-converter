"""仕様: PageMarkdown リストと元PDFから OCR オーバーレイ PDF を生成するエクスポーター。

処理内容:
- pypdfium2 で元PDF各ページを画像レンダリングし、reportlab の背景として配置する
- Yomitoku JSON の word bbox を使って不可視テキストレイヤー（rendering mode 3）を重ねる
- テキストレイヤーにより PDF の検索・コピーが可能になる（Adobe Acrobat OCR 相当）
- 単語は読み順（text_layer.load_ordered_words）に描画する。コピー・抽出したテキストの行順を本文と揃えるため
- 各単語の文字列は、水平方向に伸縮して単語 bbox の幅に収める（ページ外へのはみ出しを防ぐ）
- word_bbox_dir を指定した場合（Gemini バックエンド、issue #42）は、単語 bbox を word_bbox_dir の
  yomitoku ページキャッシュ（page_NNNN/analysis.json）から読み、文字は PageMarkdown.markdown
  （人手修正・後処理の適用後）を text_layer.assign_text で単語に割り当てたものを描画する。
  対象ページの JSON が 1 つでも無ければ、PDF を出力する前に Fail-Fast する
- 単語 bbox の JSON が無いページは背景画像のみを配置する（クラッシュしない）
- 全ページで単語 bbox が 1 件も無い場合は、テキストの無い PDF を出力せず Fail-Fast する
  （word_bbox_dir を指定しない Gemini バックエンドの出力は単語 bbox を持たないため、この経路で検出される）
- reportlab は optional dep (pdf extra) のため、未インストール時は Fail-Fast で案内する

座標変換:
  PDF座標系: 原点=左下・y軸上向き・単位=ポイント(1/72インチ)
  画像座標系: 原点=左上・y軸下向き・単位=ピクセル
  pts_per_px = 72 / dpi
  pdf_x = x1_px * pts_per_px
  pdf_y = page_height_pts - y2_px * pts_per_px  (bbox下端をPDF y座標に変換)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ouj_notebook_converter.exporters.text_layer import (
    AlignmentStats,
    OcrWord,
    assign_text,
    load_ordered_words,
    markdown_to_search_text,
)
from ouj_notebook_converter.pipeline.types import PageMarkdown

# 不可視テキストレンダリングモード（PDF仕様: テキストをストリームに含めるが描画しない）
_TEXT_RENDER_MODE_INVISIBLE = 3
# 不可視テキストのフォントサイズ下限（小さすぎると文字が欠落するリーダーがある）
_MIN_FONT_SIZE = 4.0
# 日本語 CID フォント（reportlab 同梱）
_CID_FONT_NAME = "HeiseiKakuGo-W5"
# 水平伸縮率（reportlab の setHorizScale）は百分率で指定する
_PERCENT = 100.0


def export_pdf(
    pages: list[PageMarkdown],
    out_path: Path,
    assets_dir: Path,
    *,
    source_pdf: Path,
    dpi: int = 200,
    word_bbox_dir: Path | None = None,
) -> AlignmentStats | None:
    """PageMarkdown リストと元PDFから OCR オーバーレイ PDF を書き出す。

    各ページは元PDF画像を背景に、Yomitoku JSON の word bbox 位置に
    不可視テキストを配置した searchable PDF を生成する。

    Args:
        pages: ページ順に並んだ PageMarkdown のリスト。
        out_path: 出力 .pdf ファイルのパス。
        assets_dir: インターフェース統一のために保持するが本エクスポーターでは未使用。
        source_pdf: 背景画像に使用する元 PDF ファイルのパス。
        dpi: ページ画像のレンダリング DPI。OCR 時と同じ値を使用すること。
        word_bbox_dir: 単語 bbox を読む yomitoku のページキャッシュのディレクトリ
            （page_NNNN/analysis.json を含む）。指定すると、文字は PageMarkdown.markdown から取る。
            yomitoku の OCR も同じ DPI で行ったものであること。

    Returns:
        word_bbox_dir を指定した場合は、全ページの単語への文字の割り当て件数。指定しない場合は None。

    Raises:
        ValueError: pages が空の場合、または全ページで単語 bbox が 1 件も無い場合。
        FileNotFoundError: word_bbox_dir に対象ページの analysis.json が無い場合。
        ImportError: reportlab がインストールされていない場合。
    """
    if not pages:
        raise ValueError("pages が空です。変換対象のページが存在しません。")
    if dpi <= 0:
        raise ValueError("dpi は 1 以上である必要があります。")
    word_json_paths: list[Path | None]
    if word_bbox_dir is None:
        word_json_paths = [pm.yomitoku_json_path for pm in pages]
    else:
        missing = find_missing_word_bbox_pages(word_bbox_dir, [pm.page_index for pm in pages])
        if missing:
            raise FileNotFoundError(
                f"単語 bbox の JSON（yomitoku のページキャッシュ）が {word_bbox_dir} にありません: "
                f"{', '.join(f'{p} ページ' for p in missing)}"
            )
        word_json_paths = [word_bbox_json_path(word_bbox_dir, pm.page_index) for pm in pages]
    if not any(_has_words(p) for p in word_json_paths):
        raise ValueError(
            "全ページで OCR 結果の単語 bbox が見つかりません。"
            "テキストレイヤーを作れないため searchable PDF を出力できません。"
            "単語 bbox を出力する OCR バックエンド（yomitoku）で変換してください。"
        )

    try:
        import pypdfium2
        from reportlab.lib.pagesizes import A4  # noqa: F401 — 型確認用インポート
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError as e:
        raise ImportError(
            "reportlab がインストールされていません。\n"
            "次のコマンドでインストールしてください:\n"
            "  uv sync --extra pdf\n"
            "  または: pip install 'ouj-notebook-converter[pdf]'"
        ) from e

    # 日本語 CID フォントを登録する（二重登録は無害）
    pdfmetrics.registerFont(UnicodeCIDFont(_CID_FONT_NAME))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_doc = pypdfium2.PdfDocument(str(source_pdf))

    # reportlab Canvas の pagesize は最初のページで決定する。
    # ページごとに異なるサイズに対応するため各ページで setPageSize を呼ぶ。
    c = rl_canvas.Canvas(str(out_path))
    total_stats: AlignmentStats | None = None

    try:
        for pm, json_path in zip(pages, word_json_paths, strict=True):
            page_idx = pm.page_index
            if page_idx < len(src_doc):
                src_page = src_doc[page_idx]
            else:
                # source_pdf にページが存在しない場合は最終ページで代替する
                src_page = src_doc[len(src_doc) - 1]

            # 元PDFのページサイズ（ポイント単位）を取得する
            page_width_pts = src_page.get_width()
            page_height_pts = src_page.get_height()

            c.setPageSize((page_width_pts, page_height_pts))

            # ページを画像としてレンダリングする（DPI スケール）
            scale = dpi / 72.0
            bitmap = src_page.render(scale=scale)
            pil_image = bitmap.to_pil()
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # 背景として元ページ画像を配置する（ページ全体を覆う）
            c.drawImage(
                ImageReader(img_bytes),
                0,
                0,
                width=page_width_pts,
                height=page_height_pts,
            )

            # 単語 bbox の JSON が存在する場合、不可視テキストレイヤーを配置する
            if json_path is not None and json_path.exists():
                words = load_ordered_words(json_path)
                if word_bbox_dir is None:
                    texts = [w.text for w in words]
                else:
                    assignment = assign_text(
                        [w.text for w in words], markdown_to_search_text(pm.markdown)
                    )
                    texts = list(assignment.texts)
                    total_stats = (
                        assignment.stats if total_stats is None else total_stats + assignment.stats
                    )
                _draw_invisible_text(c, words, texts, page_height_pts, dpi)

            c.showPage()

        c.save()
    finally:
        src_doc.close()
    return total_stats


def word_bbox_json_path(word_bbox_dir: Path, page_index: int) -> Path:
    """yomitoku のページキャッシュから、0 始まりの page_index のページの analysis.json のパスを返す。"""
    return word_bbox_dir / f"page_{page_index + 1:04d}" / "analysis.json"


def find_missing_word_bbox_pages(word_bbox_dir: Path, page_indices: list[int]) -> list[int]:
    """word_bbox_dir に analysis.json が無いページの番号（1 始まり）を返す。

    Args:
        word_bbox_dir: yomitoku のページキャッシュのディレクトリ。
        page_indices: 確認するページ（0 始まり）。
    """
    return [i + 1 for i in page_indices if not word_bbox_json_path(word_bbox_dir, i).is_file()]


def _has_words(json_path: Path | None) -> bool:
    """OCR 結果 JSON に不可視テキストとして描画できる単語が 1 件以上あるかを判定する。

    Args:
        json_path: OCR 結果の JSON ファイルパス。None または存在しない場合は False。

    Returns:
        content と points を両方持つ単語が 1 件以上あれば True。
    """
    if json_path is None or not json_path.exists():
        return False
    return bool(load_ordered_words(json_path))


def _draw_invisible_text(
    c: Any,
    words: list[OcrWord],
    texts: list[str],
    page_height_pts: float,
    dpi: int,
) -> None:
    """単語 bbox の位置に、対応する文字列を不可視テキストとして PDF に描画する。

    文字列は bbox の幅に合わせて水平方向に伸縮する（文字数が bbox と合わない場合も
    ページ内に収まり、選択範囲が単語の位置と揃う）。空の文字列は描画しない。

    Args:
        c: reportlab Canvas オブジェクト。
        words: 読み順の単語。
        texts: words[i] の位置に描画する文字列。
        page_height_pts: PDF ページの高さ（ポイント単位）。y座標反転に使用。
        dpi: OCR 時のレンダリング DPI。ピクセル→ポイント変換に使用。
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    pts_per_px = 72.0 / dpi

    c.saveState()

    for word, content in zip(words, texts, strict=True):
        if not content:
            continue
        x1_px, y1_px, x2_px, y2_px = word.bbox

        # ピクセル座標 → PDF ポイント座標に変換する
        pdf_x = x1_px * pts_per_px
        bbox_w_pts = (x2_px - x1_px) * pts_per_px
        bbox_h_pts = (y2_px - y1_px) * pts_per_px
        # PDF y座標は bbox の下端（画像では y2 が下端）
        pdf_y = page_height_pts - y2_px * pts_per_px

        # bbox 高さからフォントサイズを推定する（最低 _MIN_FONT_SIZE ポイントを保証）
        font_size = max(_MIN_FONT_SIZE, bbox_h_pts)
        text_width = stringWidth(content, _CID_FONT_NAME, font_size)

        # textObject 経由で不可視テキストレンダリングモードを設定する
        to = c.beginText(pdf_x, pdf_y)
        to.setTextRenderMode(_TEXT_RENDER_MODE_INVISIBLE)
        to.setFont(_CID_FONT_NAME, font_size)
        if text_width > 0 and bbox_w_pts > 0:
            to.setHorizScale(bbox_w_pts / text_width * _PERCENT)
        to.textLine(content)
        c.drawText(to)

    c.restoreState()
