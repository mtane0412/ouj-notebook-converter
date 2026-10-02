"""仕様: detectors.scoring モジュールのユニットテスト。

検出器が誤読候補にした数式を評価セットの正解と照らし合わせ、
適合率（誤読候補のうち本当に誤読だった割合）と再現率（誤読のうち検出できた割合）を数える。
"""

from ouj_notebook_converter.detectors.scoring import DetectionScore, score_page, sum_scores


class TestScorePage:
    """score_page: 1 ページ分の誤読候補を正解と照らし合わせる。"""

    def test_counts_overlap_between_flagged_and_misread_formulas(self) -> None:
        """正解と一致しない数式を誤読とし、誤読候補との重なりを数える。"""
        truth = "$x = 1$ と $(\\sqrt[4]{a})^4 = a$ と $y = 2$"
        # 2 番目（4 乗根を 3 乗根と誤読）と 3 番目（2 を 3 と誤読）の 2 式が正解と一致しない
        pred = "$x = 1$ と $(\\sqrt[3]{a})^4 = a$ と $y = 3$"
        # 1 番目（正しい式）と 2 番目（誤読）を誤読候補にした
        score = score_page(truth, pred, flagged_formula_indices={0, 1})
        assert score == DetectionScore(flagged=2, true_positive=1, misread=2)
        assert score.precision == 0.5
        assert score.recall == 0.5

    def test_notation_only_difference_is_not_misread(self) -> None:
        """\\dfrac と \\frac、$ と $$ のような表記の違いだけの数式は誤読として数えない。"""
        truth = "$\\dfrac{1}{2} = 0.5$"
        pred = "$$\\frac{1}{2} = 0.5$$"
        assert score_page(truth, pred, flagged_formula_indices=set()) == DetectionScore(
            flagged=0, true_positive=0, misread=0
        )

    def test_precision_and_recall_are_one_when_nothing_to_count(self) -> None:
        """誤読候補も誤読も無ければ、適合率と再現率を 1 とする。"""
        score = DetectionScore(flagged=0, true_positive=0, misread=0)
        assert score.precision == 1.0
        assert score.recall == 1.0


def test_sum_scores_adds_counts() -> None:
    """sum_scores は各件数を合計する。"""
    total = sum_scores(
        [
            DetectionScore(flagged=2, true_positive=2, misread=4),
            DetectionScore(flagged=1, true_positive=0, misread=1),
        ]
    )
    assert total == DetectionScore(flagged=3, true_positive=2, misread=5)
