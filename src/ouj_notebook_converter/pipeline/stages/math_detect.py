"""仕様: Pix2Text HTTP API で数式を検出・認識し、yomitoku paragraph と IoA マッチして MathOverlay を構築するステージ。

処理フロー:
  1. ページ画像を PNG として保存（Pix2Text HTTP API に送るため）
  2. detector.detect_and_recognize() でページ全体から数式を検出・LaTeX 化
  3. yomitoku words の日本語 word を除外して detection.box をトリミング（日本語ラベル混入対策）
  4. トリミングした場合は recognizer.recognize_image() で再認識して LaTeX を差し替え
  5. analysis.json の全 paragraph から IoA 最大の paragraph とマッチング
  6. マッチした paragraph の contents を original として MathOverlay に登録
  7. マッチしない検出 / 全体が日本語 word の検出は warning ログを出してスキップ

IoA マッチ閾値: 0.5
  IoA = intersection / formula_bbox_area
  インライン数式は paragraph bbox に完全包含されるが IoU は極小になるため、
  数式 bbox が paragraph に占める割合（IoA）でマッチングする。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from yomitoku.utils.misc import save_image

from ouj_notebook_converter.pipeline.stages.math_extract import MathParagraph, crop_math_image
from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis
from ouj_notebook_converter.plugins.math.base import (
    FormulaDetection,
    MathDetectorProtocol,
    MathRecognizerProtocol,
)

logger = logging.getLogger(__name__)

_TYPE_TO_ROLE = {"embedding": "inline_formula", "isolated": "display_formula"}
_IOA_THRESHOLD = 0.5

# CJK 統合漢字 + ひらがな + カタカナ（数式記号 ∑ ≤ ∈ や ASCII は範囲外）
_JAPANESE_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")


def is_japanese_text(content: str) -> bool:
    """content にひらがな・カタカナ・CJK 漢字が含まれるかを返す。

    ASCII 英数字、数式記号（∑、≤、∈ 等）、空文字列は False を返す。

    Args:
        content: 判定する文字列。

    Returns:
        日本語文字を 1 文字以上含む場合 True、そうでなければ False。
    """
    return bool(_JAPANESE_RE.search(content))


def word_points_to_box(points: list[list[int]]) -> tuple[int, int, int, int]:
    """yomitoku word の 4 点 polygon を axis-aligned bbox (x1, y1, x2, y2) に変換する。

    Args:
        points: [[x, y], ...] 形式の 4 点座標リスト。

    Returns:
        (x1, y1, x2, y2) のタプル。x1 ≤ x2、y1 ≤ y2 が保証される。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def trim_bbox_by_japanese_words(
    formula_box: tuple[int, int, int, int],
    words: list[dict[str, Any]],
    *,
    overlap_ioa_threshold: float = 0.5,
) -> tuple[int, int, int, int] | None:
    """formula_box から日本語ラベル word を除外したトリミング後 bbox を返す。

    日本語文字を含む word の領域を除外し、残った非日本語 word の最小外接矩形を
    formula_box でクリップした bbox を返す。全 word が日本語の場合は None を返す。

    Args:
        formula_box: Pix2Text が検出した数式の bbox (x1, y1, x2, y2)。
        words: yomitoku analysis.json の words[] リスト。各 word は
               "points" (list[list[int]]) と "content" (str) を持つ。
        overlap_ioa_threshold: word の IoA（word_area に対する formula との交差比）の
               閾値。この値以上重なる word のみを処理対象とする。

    Returns:
        トリミング後の bbox (x1, y1, x2, y2)。words が空または日本語 word が
        存在しない場合は formula_box をそのまま返す。全体が日本語 word の場合は None。
        トリミング結果の幅または高さが 0 以下の場合は None。
    """
    if not words:
        return formula_box

    fx1, fy1, fx2, fy2 = formula_box
    formula_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
    if formula_area == 0:
        return None

    non_japanese_words: list[tuple[int, int, int, int]] = []
    has_japanese = False

    for word in words:
        content = word.get("content", "")
        pts = word.get("points", [])
        if not pts:
            continue
        wbox = word_points_to_box(pts)

        # formula_box との IoA（word_area 分母）で重なりを確認
        wx1, wy1, wx2, wy2 = wbox
        inter_x1 = max(wx1, fx1)
        inter_y1 = max(wy1, fy1)
        inter_x2 = min(wx2, fx2)
        inter_y2 = min(wy2, fy2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        word_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
        if word_area == 0:
            continue
        word_ioa = intersection / word_area
        if word_ioa < overlap_ioa_threshold:
            # formula_box とほぼ重ならない word は処理対象外
            continue

        if is_japanese_text(content):
            has_japanese = True
        else:
            non_japanese_words.append(wbox)

    if not has_japanese:
        # 日本語 word が formula_box 内に存在しない → トリミング不要
        return formula_box

    if not non_japanese_words:
        # 全 word が日本語 → 数式なしと見なす
        return None

    # 非日本語 word の最小外接矩形を formula_box でクリップ
    nx1 = max(min(b[0] for b in non_japanese_words), fx1)
    ny1 = max(min(b[1] for b in non_japanese_words), fy1)
    nx2 = min(max(b[2] for b in non_japanese_words), fx2)
    ny2 = min(max(b[3] for b in non_japanese_words), fy2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return (nx1, ny1, nx2, ny2)


def _save_trimmed_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    output_dir: Path,
    idx: int,
) -> Path:
    """トリミング後 bbox を PNG として保存し、パスを返す。

    Args:
        image: BGR ndarray（ページ全体の画像）。
        box: クロップ範囲 (x1, y1, x2, y2)。
        output_dir: 保存先ディレクトリ（存在しない場合は作成する）。
        idx: 保存ファイル名のインデックス（{idx:04d}_trimmed.png）。

    Returns:
        保存先の Path。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    # BGR → RGB に変換して PIL で保存
    rgb = image[y1:y2, x1:x2, ::-1]
    out_path = output_dir / f"{idx:04d}_trimmed.png"
    Image.fromarray(rgb).save(str(out_path))
    return out_path


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


def ioa(
    formula_box: tuple[int, int, int, int], para_box: tuple[int, int, int, int]
) -> float:
    """数式 bbox が paragraph bbox にどれだけ含まれるか（IoA: Intersection over formula Area）を返す。

    インライン数式は paragraph bbox に完全包含されるが、paragraph が大きいため
    IoU は極小になる。IoA = intersection / formula_area を使うことで
    「数式が paragraph の中にある」ことを正しく捉えられる。

    Args:
        formula_box: 数式の bbox (x1, y1, x2, y2)。
        para_box: paragraph の bbox (x1, y1, x2, y2)。

    Returns:
        IoA 値 (0.0〜1.0)。数式面積が 0 の場合は 0.0 を返す。
    """
    fx1, fy1, fx2, fy2 = formula_box
    px1, py1, px2, py2 = para_box

    inter_x1 = max(fx1, px1)
    inter_y1 = max(fy1, py1)
    inter_x2 = min(fx2, px2)
    inter_y2 = min(fy2, py2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    formula_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
    if formula_area == 0:
        return 0.0
    return intersection / formula_area


def match_paragraph_by_ioa(
    detection_box: tuple[int, int, int, int],
    paragraphs: list[dict[str, Any]],
    *,
    threshold: float = _IOA_THRESHOLD,
) -> dict[str, Any] | None:
    """IoA 最大の paragraph を返す。閾値未満なら None。

    contents が None の paragraph はスキップする（raw.md 上で置換対象がないため）。

    Args:
        detection_box: Pix2Text が検出した数式の bbox (x1, y1, x2, y2)。
        paragraphs: analysis.json の paragraphs[] リスト。
        threshold: IoA 閾値（この値未満の場合は None を返す）。

    Returns:
        IoA 最大かつ閾値以上の paragraph dict。該当なしなら None。
    """
    best_para: dict[str, Any] | None = None
    best_score = threshold

    for para in paragraphs:
        if para.get("contents") is None:
            continue
        box_raw = para["box"]
        para_box = (int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3]))
        score = ioa(detection_box, para_box)
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
    recognizer: MathRecognizerProtocol,
) -> MathOverlay:
    """1 ページ分の数式検出ステージ（Pix2Text 経由）。

    Pix2Text でページ全体から数式を検出・認識し、yomitoku words で日本語ラベルを除外して
    bbox をトリミングした後、yomitoku paragraph と IoA マッチして MathOverlay を構築する。

    日本語ラベルのみで構成される検出・マッチしない検出は warning ログを出してスキップする。
    スキップされた paragraph は yomitoku OCR テキストが raw.md にそのまま残る。

    Args:
        image: BGR ndarray（ページ全体の画像）。
        analysis: analyze ステージの出力（yomitoku_json_path を使用）。
        cache_page_dir: ページキャッシュディレクトリ（page.png と math/ サブディレクトリを作成）。
        detector: MathDetectorProtocol を満たす数式検出・認識エンジン。
        recognizer: MathRecognizerProtocol を満たす再認識エンジン（bbox トリミング後に使用）。

    Returns:
        MathOverlay（数式なしの場合は全フィールドが空）。
    """
    from ouj_notebook_converter.plugins.math.base import MathEngineError

    # ページ画像を PNG として保存（Pix2Text HTTP API に送るため）
    page_png = cache_page_dir / "page.png"
    save_image(image, str(page_png))

    # analysis.json の全 paragraph と words を読み込む
    data = json.loads(analysis.yomitoku_json_path.read_text(encoding="utf-8"))
    paragraphs: list[dict[str, Any]] = data.get("paragraphs", [])
    words: list[dict[str, Any]] = data.get("words", [])

    # Pix2Text で数式を検出・認識
    detections: list[FormulaDetection] = detector.detect_and_recognize(page_png)
    if not detections:
        return MathOverlay()

    math_dir = cache_page_dir / "math"
    items: dict[Path, str] = {}
    roles: dict[Path, str] = {}
    originals: dict[Path, str] = {}

    for idx, detection in enumerate(detections):
        # embedding（インライン数式）は段落全体置換できないためスキップ
        # インライン数式の部分置換は将来の機能拡張で対応する
        if detection.type == "embedding":
            continue

        # yomitoku words で日本語ラベルを除外して bbox をトリミング
        trimmed_box = trim_bbox_by_japanese_words(detection.box, words)
        if trimmed_box is None:
            logger.warning(
                f"日本語ラベルのみのため数式登録をスキップします: box={detection.box} "
                f"latex={detection.latex[:40]!r}"
            )
            continue

        # bbox が変わった場合のみ再認識する（トリミングして日本語ラベルを除外した画像で認識し直す）
        if trimmed_box != detection.box:
            trimmed_crop_path = _save_trimmed_crop(image, trimmed_box, math_dir, idx)
            try:
                effective_latex, _ = recognizer.recognize_image(trimmed_crop_path)
            except MathEngineError as e:
                logger.warning(f"再認識に失敗したため元の LaTeX を使用します: {e}")
                effective_latex = detection.latex
            effective_box = trimmed_box
        else:
            effective_box = detection.box
            effective_latex = detection.latex

        matched_para = match_paragraph_by_ioa(effective_box, paragraphs)
        if matched_para is None:
            latex_preview = effective_latex[:40]
            logger.warning(
                f"未マッチの数式検出をスキップします: box={effective_box} latex={latex_preview!r}"
            )
            continue

        # MathParagraph アダプタを作成して crop_math_image を再利用
        pseudo_para = MathParagraph(
            index=idx,
            role=_TYPE_TO_ROLE.get(detection.type, "inline_formula"),
            box=effective_box,
            original_contents=str(matched_para["contents"]),
        )
        crop_path = crop_math_image(image, pseudo_para, math_dir)

        items[crop_path] = effective_latex
        roles[crop_path] = _TYPE_TO_ROLE.get(detection.type, "inline_formula")
        originals[crop_path] = str(matched_para["contents"])

    return MathOverlay(items=items, roles=roles, originals=originals)
