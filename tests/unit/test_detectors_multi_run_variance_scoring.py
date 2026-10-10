"""仕様: detectors.multi_run_variance.scoring モジュールのユニットテスト。

食い違い箇所と実際の誤読（正解との不一致）の重なりから、適合率・再現率を数えることをテストする。
"""

from ouj_notebook_converter.detectors.multi_run_variance.scoring import (
    find_prose_misreads,
    score_detection,
)


class TestFindProseMisreads:
    def test_replaced_character_is_a_misread(self) -> None:
        assert find_prose_misreads("著者は隈部です", "著者は服部です") == [(3, 4)]

    def test_difference_in_digits_is_also_a_misread(self) -> None:
        """この検出器は日本語以外も対象にするため、数字の違いも誤読に数える。"""
        assert find_prose_misreads("第12章です", "第13章です") == [(2, 3)]

    def test_formula_count_difference_alone_is_not_a_misread(self) -> None:
        assert find_prose_misreads("式 $x$ です", "式です") == []


class TestScoreDetection:
    def test_variance_overlapping_misread_is_a_true_positive(self) -> None:
        truth = "著者は隈部です。"
        runs = ["著者は服部です。", "著者は隈部です。", "著者は隈部です。"]
        score = score_detection(truth, runs, min_dissent=1)
        # 基準 0: 誤読 1・候補 1・当たり 1。基準 1, 2: 誤読 0、候補 1（誤検出）
        assert score.prose.flagged == 3
        assert score.prose.true_positive == 1
        assert score.prose.misread == 1
        assert score.prose.detected == 1
        assert score.prose.precision == 1 / 3
        assert score.prose.recall == 1.0

    def test_same_misread_in_every_run_is_not_detected(self) -> None:
        """全実行が同じ誤読をすると揺れないので、検出できない。"""
        truth = "著者は隈部です。"
        runs = ["著者は服部です。"] * 3
        score = score_detection(truth, runs, min_dissent=1)
        assert score.prose.flagged == 0
        assert score.prose.misread == 3
        assert score.prose.recall == 0.0

    def test_min_dissent_raises_precision(self) -> None:
        truth = "著者は隈部です。"
        runs = ["著者は服部です。", "著者は隈部です。", "著者は隈部です。"]
        score = score_detection(truth, runs, min_dissent=2)
        # 基準 0 では 2 実行が食い違う（候補 1・当たり）。基準 1, 2 では食い違う実行は 1 つだけなので候補にならない
        assert score.prose.flagged == 1
        assert score.prose.precision == 1.0

    def test_formula_variance_is_scored_per_formula(self) -> None:
        truth = "$a+b$ と $x^2$"
        runs = ["$a+b$ と $x^3$", "$a+b$ と $x^2$"]
        score = score_detection(truth, runs, min_dissent=1)
        # 基準 0: 数式 $x^3$ が候補（誤読）。基準 1: 数式 $x^2$ が候補（誤読ではない）
        assert score.formula.flagged == 2
        assert score.formula.true_positive == 1
        assert score.formula.misread == 1
        assert score.formula.recall == 1.0
