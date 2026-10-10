"""仕様: detectors.cross_ocr_diff.scoring モジュールのユニットテスト。

Gemini と正解の地の文を比べて「Gemini の地の文の誤読」を求め、
検出器の誤読候補が誤読の位置に当たったかで適合率・再現率を数える。
"""

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import Candidate
from ouj_notebook_converter.detectors.cross_ocr_diff.scoring import (
    CrossOcrScore,
    find_gemini_misreads,
    score_page,
    sum_scores,
)


def _candidate(gemini_span: tuple[int, int] | None, page: int = 3) -> Candidate:
    """位置だけが意味を持つテスト用の誤読候補を作る。"""
    return Candidate(
        page=page,
        kind="replace",
        gemini_text="",
        gemini_before="",
        gemini_after="",
        gemini_span=gemini_span,
        yomitoku_text="",
        yomitoku_before="",
        yomitoku_after="",
        bbox=(0, 0, 1, 1),
        bbox_source="paragraph",
    )


class TestFindGeminiMisreads:
    """find_gemini_misreads: 正解と食い違う Gemini の地の文の位置を求める。"""

    def test_replaced_japanese_character_is_a_misread(self) -> None:
        misreads = find_gemini_misreads("著者は隈部正博です。", "著者は服部正博です。")
        # 「服」は Gemini の地の文の 3 文字目（0 始まり）
        assert misreads == [(3, 4)]

    def test_extra_character_in_gemini_is_a_misread(self) -> None:
        assert find_gemini_misreads("無限に続く", "無制限に続く") == [(1, 2)]

    def test_missing_character_in_gemini_is_a_misread_at_insertion_point(self) -> None:
        assert find_gemini_misreads("微分積分学", "微積分学") == [(1, 1)]

    def test_difference_without_japanese_is_not_a_misread(self) -> None:
        """数字や記号だけの違い（式番号の取り込みなど）は、地の文の誤読として数えない。"""
        assert find_gemini_misreads("結合法則(1.4)乗法", "結合法則乗法") == []

    def test_identical_prose_has_no_misread(self) -> None:
        assert find_gemini_misreads("同じ文章です。", "同じ文章です。") == []


class TestScorePage:
    def test_candidate_overlapping_misread_is_true_positive(self) -> None:
        score = score_page(misreads=[(3, 4)], candidates=[_candidate((3, 4)), _candidate((10, 11))])
        assert score == CrossOcrScore(flagged=2, true_positive=1, misread=1, detected=1)
        assert score.precision == 0.5
        assert score.recall == 1.0

    def test_candidate_next_to_misread_counts_within_tolerance(self) -> None:
        """位置が 1 文字ずれていても、同じ誤読の候補として数える。"""
        score = score_page(misreads=[(5, 5)], candidates=[_candidate((6, 7))])
        assert score.true_positive == 1

    def test_candidate_far_from_misread_is_false_positive(self) -> None:
        score = score_page(misreads=[(5, 6)], candidates=[_candidate((20, 21))])
        assert score == CrossOcrScore(flagged=1, true_positive=0, misread=1, detected=0)

    def test_two_candidates_on_one_misread_detect_it_once(self) -> None:
        score = score_page(misreads=[(5, 6)], candidates=[_candidate((5, 6)), _candidate((4, 5))])
        assert score.true_positive == 2
        assert score.detected == 1
        assert score.recall == 1.0

    def test_candidate_without_gemini_position_never_hits(self) -> None:
        score = score_page(misreads=[(5, 6)], candidates=[_candidate(None)])
        assert score.true_positive == 0

    def test_precision_and_recall_are_one_when_nothing_to_count(self) -> None:
        score = CrossOcrScore(flagged=0, true_positive=0, misread=0, detected=0)
        assert score.precision == 1.0
        assert score.recall == 1.0


def test_sum_scores_adds_counts() -> None:
    total = sum_scores(
        [
            CrossOcrScore(flagged=2, true_positive=1, misread=1, detected=1),
            CrossOcrScore(flagged=3, true_positive=0, misread=2, detected=0),
        ]
    )
    assert total == CrossOcrScore(flagged=5, true_positive=1, misread=3, detected=1)
