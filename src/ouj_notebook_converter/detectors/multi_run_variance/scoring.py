"""仕様: 実行間の揺れ（不確実箇所の候補）を評価セットの正解と照らし合わせ、適合率・再現率を数える（issue #27）。

評価の前提: 揺れの検出は「ある 1 回の実行（基準）の出力のうち、誤読していそうな箇所を示す」ために使う。
そこで、各実行を順に基準にして評価し、件数を合算する（どの実行を採用しても同じ条件で測れる）。

定義（地の文）:
  - 誤読: 基準の実行の地の文と正解の地の文を文字単位で比べたときの食い違い（かな・漢字に限らない）
  - 候補が誤読に当たる: 候補の基準側の範囲が、誤読の範囲と TOLERANCE 文字以内で重なる
  - 適合率 = 誤読に当たった候補の件数 / 候補の件数
  - 再現率 = 候補が当たった誤読の件数 / 誤読の件数

定義（数式。数式 1 つを単位とする。equation_check と同じ定義）:
  - 誤読: 基準の実行の数式のうち、正解の数式と多重集合として一致しないもの
  - 候補にした数式: 候補の範囲に含まれる、基準の実行の数式
  - 適合率 = 候補にした数式のうち誤読だった件数 / 候補にした数式の件数
  - 再現率 = 候補にした数式のうち誤読だった件数 / 誤読の件数

注意事項:
  - 基準の実行に数式が無く、他の実行にだけある数式（formula_indices が空の候補）は数式の定義の
    候補にした数式に数えない（基準の出力に無い数式は誤読の定義にも入らない）
  - 数式の個数が実行間で違うだけの地の文の食い違いは、誤読にも候補にも数えない
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import MATH_MARKER, extract_gemini_prose
from ouj_notebook_converter.detectors.cross_ocr_diff.scoring import (
    TOLERANCE,
    CrossOcrScore,
)
from ouj_notebook_converter.detectors.cross_ocr_diff.scoring import (
    sum_scores as sum_prose_scores,
)
from ouj_notebook_converter.detectors.multi_run_variance.detect import (
    DetectOptions,
    VarianceCandidate,
    detect_page,
)
from ouj_notebook_converter.detectors.scoring import DetectionScore
from ouj_notebook_converter.detectors.scoring import score_page as score_formulas
from ouj_notebook_converter.detectors.scoring import sum_scores as sum_formula_scores


@dataclass(frozen=True)
class DetectionScores:
    """地の文と数式の、候補と誤読の件数。"""

    prose: CrossOcrScore
    formula: DetectionScore

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """JSON に書き出すための辞書を返す。"""
        return {"prose": self.prose.to_dict(), "formula": self.formula.to_dict()}


def find_prose_misreads(truth_markdown: str, run_markdown: str) -> list[tuple[int, int]]:
    """正解と食い違う、実行の地の文の範囲を求める。

    Returns:
        実行の地の文（extract_gemini_prose の結果）での範囲 [開始, 終了) のリスト。
        文字が足りない誤読は空の範囲（挿入位置）で表す。数式の個数の違いだけの食い違いは含めない。
    """
    truth = extract_gemini_prose(truth_markdown)
    run = extract_gemini_prose(run_markdown)
    matcher = difflib.SequenceMatcher(None, run, truth, autojunk=False)
    return [
        (r1, r2)
        for tag, r1, r2, t1, t2 in matcher.get_opcodes()
        if tag != "equal"
        and run[r1:r2].replace(MATH_MARKER, "") != truth[t1:t2].replace(MATH_MARKER, "")
    ]


def score_detection(
    truth_markdown: str, run_markdowns: Sequence[str], min_dissent: int = 1
) -> DetectionScores:
    """1 ページ分について、各実行を順に基準にした候補を正解と照らし合わせて件数を合算する。"""
    candidates_by_reference = [
        detect_page(0, run_markdowns, reference=r, options=DetectOptions(min_dissent=1))
        for r in range(len(run_markdowns))
    ]
    return score_candidates(truth_markdown, run_markdowns, candidates_by_reference, min_dissent)


def score_candidates(
    truth_markdown: str,
    run_markdowns: Sequence[str],
    candidates_by_reference: Sequence[Sequence[VarianceCandidate]],
    min_dissent: int,
) -> DetectionScores:
    """検出済みの候補（基準の実行ごと。min_dissent 1 で検出したもの）から、しきい値を変えて件数を数える。"""
    prose_scores: list[CrossOcrScore] = []
    formula_scores: list[DetectionScore] = []
    for reference, candidates in enumerate(candidates_by_reference):
        kept = [c for c in candidates if c.dissent >= min_dissent]
        misreads = find_prose_misreads(truth_markdown, run_markdowns[reference])
        prose_scores.append(_score_prose(misreads, [c for c in kept if c.granularity == "prose"]))
        flagged = {i for c in kept if c.granularity == "formula" for i in c.formula_indices}
        formula_scores.append(score_formulas(truth_markdown, run_markdowns[reference], flagged))
    return DetectionScores(
        prose=sum_prose_scores(prose_scores), formula=sum_formula_scores(formula_scores)
    )


def sum_detection_scores(scores: Sequence[DetectionScores]) -> DetectionScores:
    """複数ページ（または複数の組み合わせ）の件数を合算する。"""
    return DetectionScores(
        prose=sum_prose_scores(s.prose for s in scores),
        formula=sum_formula_scores(s.formula for s in scores),
    )


def _score_prose(
    misreads: Sequence[tuple[int, int]], candidates: Sequence[VarianceCandidate]
) -> CrossOcrScore:
    """地の文の候補を、誤読の範囲と照らし合わせる。"""
    detected: set[int] = set()
    true_positive = 0
    for candidate in candidates:
        hits = [i for i, m in enumerate(misreads) if _overlaps(candidate.reference_span, m)]
        if hits:
            true_positive += 1
            detected.update(hits)
    return CrossOcrScore(
        flagged=len(candidates),
        true_positive=true_positive,
        misread=len(misreads),
        detected=len(detected),
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """2 つの範囲が、TOLERANCE 文字広げたうえで重なるか。"""
    return a[0] - TOLERANCE < b[1] + TOLERANCE and b[0] - TOLERANCE < a[1] + TOLERANCE
