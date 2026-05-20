"""仕様: Pix2Text HTTP API で数式を検出・認識し、yomitoku paragraph と IoA マッチして MathOverlay を構築するステージ。

処理フロー:
  1. ページ画像を PNG として保存（Pix2Text HTTP API に送るため）
  2. detector.detect_and_recognize() でページ全体から数式を検出・LaTeX 化
  3. yomitoku words の日本語 word を除外して detection.box をトリミング（日本語ラベル混入対策）
  4. トリミングした場合は recognizer.recognize_image() で再認識して LaTeX を差し替え
  5. analysis.json の全 paragraph から IoA 最大の paragraph とマッチング
  6. detection.type に応じて振り分け:
     - isolated (display_formula) : paragraph.contents を originals に登録（段落全体置換）
     - embedding (inline_formula) : paragraph 内 word を IoA で収集し、embedding bbox と重なる
                                    word の span を inline_paragraphs に登録（段落内部分置換）
  7. マッチしない検出 / 全体が日本語 word の検出 / word が見つからない embedding は
     warning ログを出してスキップ（yomitoku OCR テキストがそのまま raw.md に残る）

IoA マッチ閾値: 0.5
  IoA = intersection / formula_bbox_area
  インライン数式は paragraph bbox に完全包含されるが IoU は極小になるため、
  数式 bbox が paragraph に占める割合（IoA）でマッチングする。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from yomitoku.utils.misc import save_image

from ouj_notebook_converter.pipeline.types import (
    InlineParagraphReplacement,
    MathOverlay,
    PageAnalysis,
)
from ouj_notebook_converter.plugins.math.base import (
    FormulaDetection,
    MathDetectorProtocol,
    MathRecognizerProtocol,
)

logger = logging.getLogger(__name__)

_TYPE_TO_ROLE = {"embedding": "inline_formula", "isolated": "display_formula"}


@dataclasses.dataclass(frozen=True)
class MathParagraph:
    """analysis.json から抽出した数式 paragraph の情報。"""

    index: int
    role: str
    box: tuple[int, int, int, int]
    original_contents: str


def crop_math_image(
    image: np.ndarray,
    paragraph: MathParagraph,
    output_dir: Path,
) -> Path:
    """元画像から数式 paragraph の bbox をクロップして PNG として保存する。

    Args:
        image: BGR 形式の NumPy 配列（yomitoku の load_pdf が返す形式）。
        paragraph: 抽出済みの数式 paragraph 情報。
        output_dir: クロップ画像の保存先ディレクトリ（自動作成）。

    Returns:
        保存した PNG ファイルのパス（output_dir / f"{paragraph.index:04d}.png"）。

    Raises:
        ValueError: クランプ後の bbox サイズが 0 になった場合。
    """
    x1, y1, x2, y2 = paragraph.box
    h, w = image.shape[:2]

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"数式 bbox のサイズが 0 です: box={paragraph.box}, image_size=({w}x{h})")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paragraph.index:04d}.png"

    crop = image[y1:y2, x1:x2, :]
    save_image(crop, str(output_path))

    return output_path
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
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"クランプ後に空クロップになりました: box=({x1},{y1},{x2},{y2})")
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


def collect_words_in_paragraph(
    para_box: tuple[int, int, int, int],
    words: list[dict[str, Any]],
    *,
    contain_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """paragraph bbox に IoA（word_area 分母）で閾値以上含まれる word を返す。

    yomitoku の extract_words_within_element と同じ閾値（0.5）を採用する。

    Args:
        para_box: paragraph の bbox (x1, y1, x2, y2)。
        words: analysis.json の words[] リスト。
        contain_threshold: word IoA の閾値（この値以上の word のみ返す）。

    Returns:
        paragraph 内に含まれる word dict のリスト。
    """
    px1, py1, px2, py2 = para_box
    result = []
    for word in words:
        pts = word.get("points", [])
        if not pts:
            continue
        wx1, wy1, wx2, wy2 = word_points_to_box(pts)
        word_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
        if word_area == 0:
            continue
        inter_x1 = max(wx1, px1)
        inter_y1 = max(wy1, py1)
        inter_x2 = min(wx2, px2)
        inter_y2 = min(wy2, py2)
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if intersection / word_area >= contain_threshold:
            result.append(word)
    return result


def sort_words_reading_order(
    words: list[dict[str, Any]],
    direction: str,
) -> list[dict[str, Any]]:
    """方向に応じて word を reading order でソートする純粋関数。

    horizontal: 行（y 中心でクラスタリング）ごとに上→下、行内は x 昇順（左→右）。
    vertical  : 列（x 中心でクラスタリング）ごとに右→左、列内は y 昇順（上→下）。

    Args:
        words: ソート対象の word dict リスト。
        direction: "horizontal" または "vertical"。

    Returns:
        reading order でソートされた word dict リスト（元リストは変更しない）。
    """
    if not words:
        return []

    def _center(word: dict[str, Any]) -> tuple[float, float]:
        pts = word.get("points", [])
        if pts:
            box = word_points_to_box(pts)
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return (0.0, 0.0)

    if direction == "vertical":
        # 右から左への列順（x 降順）、列内は y 昇順
        return sorted(words, key=lambda w: (-_center(w)[0], _center(w)[1]))
    else:
        # 横書き: 行順（y 昇順）、行内は x 昇順
        return sorted(words, key=lambda w: (_center(w)[1], _center(w)[0]))


def select_words_inside_embedding(
    sorted_words: list[dict[str, Any]],
    embedding_box: tuple[int, int, int, int],
    *,
    overlap_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], int]:
    """reading-order ソート済み word 列のうち embedding bbox に重なる連続範囲を返す。

    各 word の IoA（word_area 分母）が overlap_threshold 以上の word をマッチ対象とし、
    その最小〜最大インデックスの連続範囲を採用する。範囲内に IoA ゼロの word があれば
    「穴あき」とみなし空リストを返す（誤マッチ防止）。

    Args:
        sorted_words: reading order でソート済みの word dict リスト。
        embedding_box: Pix2Text が検出した embedding bbox (x1, y1, x2, y2)。
        overlap_threshold: word IoA の閾値。

    Returns:
        (selected_words, prefix_count)
        selected_words: embedding bbox 内の連続 word リスト（空の場合はスキップ）。
        prefix_count  : selected_words の先頭より前にある word の数。
    """
    ex1, ey1, ex2, ey2 = embedding_box
    embed_area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
    if embed_area == 0:
        return [], 0

    match_indices: list[int] = []
    for idx, word in enumerate(sorted_words):
        pts = word.get("points", [])
        if not pts:
            continue
        wx1, wy1, wx2, wy2 = word_points_to_box(pts)
        word_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
        if word_area == 0:
            continue
        inter_x1 = max(wx1, ex1)
        inter_y1 = max(wy1, ey1)
        inter_x2 = min(wx2, ex2)
        inter_y2 = min(wy2, ey2)
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if intersection / word_area >= overlap_threshold:
            match_indices.append(idx)

    if not match_indices:
        return [], 0

    start_idx = min(match_indices)
    end_idx = max(match_indices)
    full_range = set(range(start_idx, end_idx + 1))

    # 穴あき確認: 範囲内に IoA 閾値未満の word がある場合は空リストを返す
    if full_range != set(match_indices):
        return [], 0

    return sorted_words[start_idx : end_idx + 1], start_idx


def match_contents_to_words(
    contents: str,
    para_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """paragraph.contents の各行（= 1 word）に bbox を割り当て、reading order 済み dict リストを返す。

    yomitoku は paragraph.contents = "\\n".join(word.content for word in reading_order_words)
    で生成するため、split("\\n") の各行が yomitoku の reading order での 1 word.content に対応する。

    sort_words_reading_order より正確：行の y 座標のばらつきに依存せず、
    yomitoku が実際に採用した reading order を再現する。

    同一 content の word が複数ある場合は使用済みインデックスを追跡して順に消費する。
    対応する word が見つからない行は content のみで box なし（bbox は None 相当）の
    dict として返す（bbox なしは select_words_inside_embedding でスキップされる）。

    Args:
        contents: matched_para["contents"]（yomitoku 生成の reading order 済みテキスト）。
        para_words: paragraph 内の word dict リスト（collect_words_in_paragraph の結果）。

    Returns:
        (content, points) を持つ dict リスト（reading order 順）。
        points が取得できない場合は空リストを持つ。
    """
    lines = [line for line in contents.split("\n") if line]

    # content → word dict リスト（同一 content に複数 word がある場合の重複対応）
    content_to_words: dict[str, list[dict[str, Any]]] = {}
    for w in para_words:
        c = w.get("content", "")
        content_to_words.setdefault(c, []).append(w)

    content_used: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for line in lines:
        used_count = content_used.get(line, 0)
        candidates = content_to_words.get(line, [])
        if used_count < len(candidates):
            word = candidates[used_count]
            content_used[line] = used_count + 1
            result.append(word)
        else:
            # 対応する word が見つからない場合は content のみの空 dict（bbox なし）
            result.append({"content": line, "points": []})

    return result


def build_fragment(words: list[dict[str, Any]]) -> str:
    """word.content を順に連結した fragment 文字列を返す純粋関数。

    yomitoku は paragraph.contents = "\\n".join(word.contents) で生成し、
    raw.md 出力時に "\\n" → "" と置換するため、word.content の連結が raw.md 内の
    部分文字列に対応する。

    Args:
        words: 連結対象の word dict リスト。

    Returns:
        word.content を連結した文字列。
    """
    return "".join(w.get("content", "") for w in words)


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
    # paragraph インデックス → {word_contents, latex_spans の蓄積リスト}
    inline_para_builders: dict[int, tuple[tuple[str, ...], list[tuple[int, int, str]]]] = {}

    for idx, detection in enumerate(detections):
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
            except (MathEngineError, ValueError, OSError) as e:
                logger.warning(f"再認識に失敗したため元の LaTeX を使用します: {e}")
                effective_latex = detection.latex
                effective_box = detection.box
            else:
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

        role = _TYPE_TO_ROLE.get(detection.type, "inline_formula")

        if role == "inline_formula":
            # paragraph 単位で word を集約し、embedding bbox に対応する span を登録
            para_box_raw = matched_para["box"]
            para_box = (
                int(para_box_raw[0]),
                int(para_box_raw[1]),
                int(para_box_raw[2]),
                int(para_box_raw[3]),
            )
            para_words = collect_words_in_paragraph(para_box, words)
            # paragraph.contents の行順が yomitoku の actual reading order を正確に反映する
            # sort_words_reading_order の代わりに match_contents_to_words を使う
            # （同一行の word 間で y 座標のばらつきがあってもソート逆転が起きない）
            ordered_para_words = match_contents_to_words(
                str(matched_para.get("contents") or ""), para_words
            )
            selected, prefix_count = select_words_inside_embedding(ordered_para_words, effective_box)
            if not selected:
                logger.warning(
                    f"embedding bbox に重なる word が見つからないためスキップします: "
                    f"box={effective_box} latex={effective_latex[:40]!r}"
                )
                continue

            para_idx = paragraphs.index(matched_para)
            word_contents = tuple(w.get("content", "") for w in ordered_para_words)
            span = (prefix_count, prefix_count + len(selected), effective_latex)

            if para_idx in inline_para_builders:
                # 既存エントリに span を追加
                existing_contents, existing_spans = inline_para_builders[para_idx]
                inline_para_builders[para_idx] = (existing_contents, [*existing_spans, span])
            else:
                inline_para_builders[para_idx] = (word_contents, [span])
        else:
            # display_formula: 既存通り crop_math_image で登録
            pseudo_para = MathParagraph(
                index=idx,
                role=role,
                box=effective_box,
                original_contents=str(matched_para["contents"]),
            )
            crop_path = crop_math_image(image, pseudo_para, math_dir)
            items[crop_path] = effective_latex
            roles[crop_path] = role
            originals[crop_path] = str(matched_para["contents"])

    # inline_para_builders を InlineParagraphReplacement に変換
    inline_paragraphs: dict[int, InlineParagraphReplacement] = {
        para_idx: InlineParagraphReplacement(
            word_contents=word_contents,
            latex_spans=tuple(sorted(spans, key=lambda s: s[0])),
        )
        for para_idx, (word_contents, spans) in inline_para_builders.items()
    }

    return MathOverlay(items=items, roles=roles, originals=originals, inline_paragraphs=inline_paragraphs)
