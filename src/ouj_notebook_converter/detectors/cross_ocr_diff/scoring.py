"""仕様: 誤読候補（Gemini と yomitoku の食い違い）を評価セットの正解と照らし合わせ、適合率・再現率を数える。

定義（Gemini の地の文の誤読 1 か所を単位とする）:
  - 誤読: Gemini の地の文と正解の地の文を文字単位で比べたときの食い違いのうち、
    どちらかにかな・漢字を含むもの（数字・記号だけの違いは地の文の誤読として数えない）
  - 誤読候補が誤読に当たる: 候補の Gemini 側の位置が、誤読の位置と TOLERANCE 文字以内で重なる
  - 適合率 = 誤読に当たった候補の件数 / 候補の件数
  - 再現率 = 候補が当たった誤読の件数 / 誤読の件数

注意事項:
  - 位置は Gemini の地の文（extract_gemini_prose の結果）での文字位置。正解側にも同じ関数を使い、数式の印を揃える
  - Gemini 側の位置を持たない候補（unmatched_run）は、誤読に当たらないものとして数える
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    Candidate,
    extract_gemini_prose,
)

# 候補と誤読の位置が何文字ずれていても同じ箇所とみなすか（範囲の両端を広げる幅）
TOLERANCE = 1
# かな・漢字の範囲（長音符「ー」は含めない）
_JAPANESE_RANGES = (
    ("ぁ", "ゟ"),
    ("ァ", "ヺ"),
    ("㐀", "鿿"),
    ("々", "々"),
    ("〆", "〆"),
)


@dataclass(frozen=True)
class CrossOcrScore:
    """誤読候補と誤読の件数。

    Attributes:
        flagged: 誤読候補の件数。
        true_positive: 誤読に当たった候補の件数。
        misread: 誤読の件数。
        detected: 候補が当たった誤読の件数。
    """

    flagged: int
    true_positive: int
    misread: int
    detected: int

    @property
    def precision(self) -> float:
        """適合率。候補が 0 件なら誤検出も無いため 1 とする。"""
        return self.true_positive / self.flagged if self.flagged else 1.0

    @property
    def recall(self) -> float:
        """再現率。誤読が 0 件なら見落としも無いため 1 とする。"""
        return self.detected / self.misread if self.misread else 1.0

    def to_dict(self) -> dict[str, int | float]:
        """JSON に書き出すための辞書を返す。"""
        return {
            "flagged": self.flagged,
            "true_positive": self.true_positive,
            "misread": self.misread,
            "detected": self.detected,
            "precision": self.precision,
            "recall": self.recall,
        }


def find_gemini_misreads(truth_markdown: str, gemini_markdown: str) -> list[tuple[int, int]]:
    """正解と食い違う Gemini の地の文の位置を求める。

    Args:
        truth_markdown: 正解 Markdown。
        gemini_markdown: Gemini の Markdown（検出器に入力したもの）。

    Returns:
        Gemini の地の文（extract_gemini_prose の結果）での範囲 [開始, 終了) のリスト。
        Gemini に文字が足りない誤読は、空の範囲（挿入位置）で表す。
    """
    truth = extract_gemini_prose(truth_markdown)
    gemini = extract_gemini_prose(gemini_markdown)
    matcher = difflib.SequenceMatcher(None, gemini, truth, autojunk=False)
    return [
        (g1, g2)
        for tag, g1, g2, t1, t2 in matcher.get_opcodes()
        if tag != "equal" and _has_japanese(gemini[g1:g2] + truth[t1:t2])
    ]


def score_page(
    misreads: Sequence[tuple[int, int]], candidates: Sequence[Candidate]
) -> CrossOcrScore:
    """1 ページ分の誤読候補を、誤読の位置と照らし合わせる。"""
    detected: set[int] = set()
    true_positive = 0
    for candidate in candidates:
        if candidate.gemini_span is None:
            continue
        hits = [i for i, m in enumerate(misreads) if _overlaps(candidate.gemini_span, m)]
        if hits:
            true_positive += 1
            detected.update(hits)
    return CrossOcrScore(
        flagged=len(candidates),
        true_positive=true_positive,
        misread=len(misreads),
        detected=len(detected),
    )


def sum_scores(scores: Iterable[CrossOcrScore]) -> CrossOcrScore:
    """複数ページの件数を合計する。"""
    items = list(scores)
    return CrossOcrScore(
        flagged=sum(s.flagged for s in items),
        true_positive=sum(s.true_positive for s in items),
        misread=sum(s.misread for s in items),
        detected=sum(s.detected for s in items),
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """2 つの範囲が、TOLERANCE 文字広げたうえで重なるか。"""
    return a[0] - TOLERANCE < b[1] + TOLERANCE and b[0] - TOLERANCE < a[1] + TOLERANCE


def _has_japanese(text: str) -> bool:
    """かな・漢字（長音符を除く）を含むか。"""
    return any(lo <= ch <= hi for ch in text for lo, hi in _JAPANESE_RANGES)
