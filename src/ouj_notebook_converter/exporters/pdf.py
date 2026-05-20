"""仕様: PageMarkdown リストと元PDFから OCR オーバーレイ PDF を生成するエクスポーター。

処理内容:
- pypdfium2 で元PDF各ページを画像レンダリングし、reportlab の背景として配置する
- Yomitoku JSON の word bbox を使って不可視テキストレイヤー（rendering mode 3）を重ねる
- テキストレイヤーにより PDF の検索・コピーが可能になる（Adobe Acrobat OCR 相当）
- yomitoku_json_path が None のページは背景画像のみを配置する（クラッシュしない）
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
import json
from pathlib import Path
from typing import Any

from ouj_notebook_converter.pipeline.types import PageMarkdown

# 不可視テキストレンダリングモード（PDF仕様: テキストをストリームに含めるが描画しない）
_TEXT_RENDER_MODE_INVISIBLE = 3
# 不可視テキストのフォントサイズ下限（小さすぎると文字が欠落するリーダーがある）
_MIN_FONT_SIZE = 4.0
# 日本語 CID フォント（reportlab 同梱）
_CID_FONT_NAME = "HeiseiKakuGo-W5"


def export_pdf(
    pages: list[PageMarkdown],
    out_path: Path,
    assets_dir: Path,
    *,
    source_pdf: Path,
    dpi: int = 200,
) -> None:
    """PageMarkdown リストと元PDFから OCR オーバーレイ PDF を書き出す。

    各ページは元PDF画像を背景に、Yomitoku JSON の word bbox 位置に
    不可視テキストを配置した searchable PDF を生成する。

    Args:
        pages: ページ順に並んだ PageMarkdown のリスト。
        out_path: 出力 .pdf ファイルのパス。
        assets_dir: インターフェース統一のために保持するが本エクスポーターでは未使用。
        source_pdf: 背景画像に使用する元 PDF ファイルのパス。
        dpi: ページ画像のレンダリング DPI。OCR 時と同じ値を使用すること。

    Raises:
        ValueError: pages が空の場合。
        ImportError: reportlab がインストールされていない場合。
    """
    if not pages:
        raise ValueError("pages が空です。変換対象のページが存在しません。")
    if dpi <= 0:
        raise ValueError("dpi は 1 以上である必要があります。")

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

    try:
        for pm in pages:
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

            # Yomitoku JSON が存在する場合、不可視テキストレイヤーを配置する
            if pm.yomitoku_json_path is not None and pm.yomitoku_json_path.exists():
                _draw_invisible_text(c, pm.yomitoku_json_path, page_height_pts, dpi)

            c.showPage()

        c.save()
    finally:
        src_doc.close()


def _draw_invisible_text(
    c: Any,
    json_path: Path,
    page_height_pts: float,
    dpi: int,
) -> None:
    """Yomitoku JSON の word bbox を読み込み、不可視テキストを PDF に描画する。

    Args:
        c: reportlab Canvas オブジェクト。
        json_path: Yomitoku OCR 結果の JSON ファイルパス。
        page_height_pts: PDF ページの高さ（ポイント単位）。y座標反転に使用。
        dpi: OCR 時のレンダリング DPI。ピクセル→ポイント変換に使用。
    """
    pts_per_px = 72.0 / dpi
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    words: list[dict[str, Any]] = data.get("words", [])

    c.saveState()

    for word in words:
        content: str = word.get("content", "")
        points: list[list[int]] = word.get("points", [])
        if not content or not points:
            continue

        # 4 点ポリゴン → axis-aligned bbox (x1, y1, x2, y2) に変換する
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1_px = min(xs)
        y2_px = max(ys)
        y1_px = min(ys)

        # ピクセル座標 → PDF ポイント座標に変換する
        pdf_x = x1_px * pts_per_px
        bbox_h_pts = (y2_px - y1_px) * pts_per_px
        # PDF y座標は bbox の下端（画像では y2 が下端）
        pdf_y = page_height_pts - y2_px * pts_per_px

        # bbox 高さからフォントサイズを推定する（最低 _MIN_FONT_SIZE ポイントを保証）
        font_size = max(_MIN_FONT_SIZE, bbox_h_pts)

        # textObject 経由で不可視テキストレンダリングモードを設定する
        to = c.beginText(pdf_x, pdf_y)
        to.setTextRenderMode(_TEXT_RENDER_MODE_INVISIBLE)
        to.setFont(_CID_FONT_NAME, font_size)
        to.textLine(content)
        c.drawText(to)

    c.restoreState()
