"""仕様: Gemini の出力（bbox なし）の候補が、原本ページ画像のどこにあるかを推定する。

方針（上から順に試す）:
  1. 検出器が yomitoku の bbox を持っていればそれを使う（cross_ocr_diff）
  2. line_alignment: 候補を含む Markdown の行（日本語 6 文字以上）を、日本語部分の文字 2-gram の
     Dice 係数が最も高い yomitoku の段落に対応付ける
  3. band_between: 数式だけの行は日本語が無く対応付けできない。上下の最も近い「対応付けできた行」の
     段落に挟まれた帯（横幅は 2 つの段落の和集合）を位置とする。片側しか無ければ、その段落から
     一定の高さ分だけ反対側へ伸ばす
  4. 特定できなければ bbox なし（ページ全体を画像として渡す）

座標は yomitoku の analysis.json と同じ 200 DPI のピクセル座標。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    _Analysis,
    extract_gemini_prose,
)
from ouj_notebook_converter.judge.models import BBox, JudgeCandidate, LocationMethod

_JAPANESE = re.compile(r"[ぁ-ゖァ-ヺー一-鿿々]")
# 対応付けの対象にする行の日本語の最小文字数（短い行は偶然の一致が多いため）
MIN_LINE_CHARS = 6
# 対応付けとみなす文字 2-gram の Dice 係数の下限
MIN_DICE = 0.5
# 片側にしか対応付けできた段落が無い数式行について、段落から伸ばす高さ（px, 200 DPI）
ONE_SIDED_BAND_HEIGHT = 300


def _japanese_only(text: str) -> str:
    """NFKC 正規化して、かな・漢字・長音符だけを残す。"""
    return "".join(_JAPANESE.findall(unicodedata.normalize("NFKC", text)))


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def align_lines(lines: list[str], analysis: _Analysis) -> dict[int, BBox]:
    """日本語を含む行 → 最も似ている yomitoku の段落の範囲、の対応表を作る。"""
    paragraphs = [(p.box, _bigrams(_japanese_only(p.contents))) for p in analysis.paragraphs]
    aligned: dict[int, BBox] = {}
    for index, line in enumerate(lines):
        text = _japanese_only(extract_gemini_prose(line))
        if len(text) < MIN_LINE_CHARS:
            continue
        grams = _bigrams(text)
        best_box, best_score = None, 0.0
        for box, other in paragraphs:
            score = _dice(grams, other)
            if score > best_score:
                best_box, best_score = box, score
        if best_box is not None and best_score >= MIN_DICE:
            aligned[index] = best_box
    return aligned


def locate_line(
    lines: list[str], line_index: int, analysis: _Analysis
) -> tuple[BBox, LocationMethod] | None:
    """lines[line_index] の原本画像上の範囲を推定する。特定できなければ None。"""
    aligned = align_lines(lines, analysis)
    if line_index in aligned:
        return aligned[line_index], "line_alignment"

    before = [i for i in aligned if i < line_index]
    after = [i for i in aligned if i > line_index]
    upper = aligned[max(before)] if before else None
    lower = aligned[min(after)] if after else None
    if upper is None and lower is None:
        return None
    if upper is not None and lower is not None:
        top, bottom = upper[3], lower[1]
        if top >= bottom:  # 読む順序が入れ替わっているなど。2 つの段落の外接で近似する
            top, bottom = min(upper[1], lower[1]), max(upper[3], lower[3])
        return (
            (min(upper[0], lower[0]), top, max(upper[2], lower[2]), bottom),
            "band_between",
        )
    if upper is not None:
        return (upper[0], upper[3], upper[2], upper[3] + ONE_SIDED_BAND_HEIGHT), "band_between"
    assert lower is not None
    return (lower[0], lower[1] - ONE_SIDED_BAND_HEIGHT, lower[2], lower[1]), "band_between"


def locate_candidate(candidate: JudgeCandidate, raw: str, analysis: _Analysis) -> JudgeCandidate:
    """候補の位置を決めて返す。

    - bbox が無ければ、Gemini の行を yomitoku の段落に対応付けて推定した位置を書き込む
    - bbox がある（yomitoku の座標）場合は、Gemini の行の対応先の段落の中に bbox の中心があれば
      そのまま使う。外れていれば、yomitoku 側の対応の取り違えとみなし、対応先の段落に置き換える。
      Gemini の行が段落に対応付けできない（数式だけの行など）ときは、bbox を変えない
    """
    if candidate.line_index is None:
        return candidate
    lines = raw.split("\n")
    if candidate.bbox is None:
        found = locate_line(lines, candidate.line_index, analysis)
        if found is None:
            return candidate
        bbox, method = found
        return replace(candidate, bbox=bbox, location=method)

    aligned = align_lines(lines, analysis).get(candidate.line_index)
    if aligned is None:
        return candidate
    center_x = (candidate.bbox[0] + candidate.bbox[2]) / 2
    center_y = (candidate.bbox[1] + candidate.bbox[3]) / 2
    if aligned[0] <= center_x <= aligned[2] and aligned[1] <= center_y <= aligned[3]:
        return candidate
    return replace(candidate, bbox=aligned, location="line_alignment")
