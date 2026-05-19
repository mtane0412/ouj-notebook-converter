"""仕様: Pix2Text HTTP API で数式を検出・認識し、yomitoku paragraph と IoU マッチして MathOverlay を構築するステージ。

処理フロー:
  1. ページ画像を PNG として保存（Pix2Text HTTP API に送るため）
  2. detector.detect_and_recognize() でページ全体から数式を検出・LaTeX 化
  3. analysis.json の全 paragraph から IoU 最大の paragraph とマッチング
  4. マッチした paragraph の contents を original として MathOverlay に登録
  5. マッチしない検出は warning ログを出してスキップ（Fail-Fast なし）

IoU マッチ閾値: 0.3（yomitoku の paragraph bbox が大きめに取られるため低めに設定）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from yomitoku.utils.misc import save_image

from ouj_notebook_converter.pipeline.stages.math_extract import MathParagraph, crop_math_image
from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis
from ouj_notebook_converter.plugins.math.base import FormulaDetection, MathDetectorProtocol

logger = logging.getLogger(__name__)

_TYPE_TO_ROLE = {"embedding": "inline_formula", "isolated": "display_formula"}
_IOU_THRESHOLD = 0.3


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """2 つの bbox の IoU (Intersection over Union) を返す純粋関数。

    Args:
        box_a: (x1, y1, x2, y2) 形式の bbox。
        box_b: (x1, y1, x2, y2) 形式の bbox。

    Returns:
        IoU 値 (0.0〜1.0)。どちらかの面積が 0 の場合は 0.0 を返す。
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return intersection / union


def match_paragraph_by_iou(
    detection_box: tuple[int, int, int, int],
    paragraphs: list[dict[str, Any]],
    *,
    threshold: float = _IOU_THRESHOLD,
) -> dict[str, Any] | None:
    """IoU 最大の paragraph を返す。閾値未満なら None。

    contents が None の paragraph はスキップする（raw.md 上で置換対象がないため）。

    Args:
        detection_box: Pix2Text が検出した数式の bbox (x1, y1, x2, y2)。
        paragraphs: analysis.json の paragraphs[] リスト。
        threshold: IoU 閾値（この値未満の場合は None を返す）。

    Returns:
        IoU 最大かつ閾値以上の paragraph dict。該当なしなら None。
    """
    best_para: dict[str, Any] | None = None
    best_score = threshold

    for para in paragraphs:
        if para.get("contents") is None:
            continue
        box_raw = para["box"]
        para_box = (int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3]))
        score = iou(detection_box, para_box)
        if score >= best_score:
            best_score = score
            best_para = para

    return best_para


def math_detect(
    image: np.ndarray,
    analysis: PageAnalysis,
    cache_page_dir: Path,
    *,
    detector: MathDetectorProtocol,
) -> MathOverlay:
    """1 ページ分の数式検出ステージ（Pix2Text 経由）。

    Pix2Text でページ全体から数式を検出・認識し、yomitoku paragraph と IoU マッチして
    MathOverlay を構築する。マッチしない検出は warning ログを出してスキップする。

    Args:
        image: BGR ndarray（ページ全体の画像）。
        analysis: analyze ステージの出力（yomitoku_json_path を使用）。
        cache_page_dir: ページキャッシュディレクトリ（page.png と math/ サブディレクトリを作成）。
        detector: MathDetectorProtocol を満たす数式検出・認識エンジン。

    Returns:
        MathOverlay（数式なしの場合は全フィールドが空）。
    """
    # ページ画像を PNG として保存（Pix2Text HTTP API に送るため）
    page_png = cache_page_dir / "page.png"
    save_image(image, str(page_png))

    # analysis.json の全 paragraph を読み込む
    data = json.loads(analysis.yomitoku_json_path.read_text(encoding="utf-8"))
    paragraphs: list[dict[str, Any]] = data.get("paragraphs", [])

    # Pix2Text で数式を検出・認識
    detections: list[FormulaDetection] = detector.detect_and_recognize(page_png)
    if not detections:
        return MathOverlay()

    math_dir = cache_page_dir / "math"
    items: dict[Path, str] = {}
    roles: dict[Path, str] = {}
    originals: dict[Path, str] = {}

    for idx, detection in enumerate(detections):
        matched_para = match_paragraph_by_iou(detection.box, paragraphs)
        if matched_para is None:
            latex_preview = detection.latex[:40]
            logger.warning(
                f"未マッチの数式検出をスキップします: box={detection.box} latex={latex_preview!r}"
            )
            continue

        # MathParagraph アダプタを作成して crop_math_image を再利用
        pseudo_para = MathParagraph(
            index=idx,
            role=_TYPE_TO_ROLE.get(detection.type, "inline_formula"),
            box=detection.box,
            original_contents=str(matched_para["contents"]),
        )
        crop_path = crop_math_image(image, pseudo_para, math_dir)

        items[crop_path] = detection.latex
        roles[crop_path] = _TYPE_TO_ROLE.get(detection.type, "inline_formula")
        originals[crop_path] = str(matched_para["contents"])

    return MathOverlay(items=items, roles=roles, originals=originals)
