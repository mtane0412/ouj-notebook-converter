"""仕様: evaluation.metrics モジュール（精度指標の計算）のユニットテスト。

編集距離・文字誤り率（CER）・多重集合の一致率（数式・見出し）と、
ページ単位の評価 evaluate_page・全体集計 summarize をテストする。
"""

import pytest

from ouj_notebook_converter.evaluation.metrics import (
    CerScore,
    MatchScore,
    evaluate_page,
    levenshtein,
    match_multiset,
    summarize,
)


class TestLevenshtein:
    """編集距離。"""

    @pytest.mark.parametrize(
        ("truth", "pred", "expected"),
        [
            ("無限に", "無限に", 0),
            ("無限に", "無制限に", 1),  # 挿入
            ("しよう", "しとう", 1),  # 置換
            ("初歩からの数学", "初歩の数学", 2),  # 削除
            ("", "abc", 3),
        ],
    )
    def test_counts_minimum_edits(self, truth: str, pred: str, expected: int) -> None:
        """挿入・削除・置換の最小回数を返す。"""
        assert levenshtein(truth, pred) == expected


class TestCerScore:
    """文字誤り率。"""

    def test_divides_edits_by_truth_length(self) -> None:
        """編集回数を正解の文字数で割る。"""
        assert CerScore(edits=1, truth_length=4).cer == 0.25

    def test_empty_truth_uses_one_as_denominator(self) -> None:
        """正解が空のページは分母を 1 とし、ゼロ除算しない。"""
        assert CerScore(edits=2, truth_length=0).cer == 2.0


class TestMatchMultiset:
    """多重集合としての一致率。"""

    def test_counts_duplicates_up_to_truth_count(self) -> None:
        """同じ要素は正解に含まれる個数までしか一致と数えない。"""
        score = match_multiset(["x", "x", "y"], ["x", "x", "x", "z"])

        assert score == MatchScore(matched=2, truth_count=3, pred_count=4)
        assert score.precision == 0.5
        assert score.recall == pytest.approx(2 / 3)
        assert score.f1 == pytest.approx(4 / 7)

    def test_empty_both_is_perfect(self) -> None:
        """正解も出力も空なら適合率・再現率・F1 はすべて 1 とする。"""
        score = match_multiset([], [])

        assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)

    def test_no_match_gives_zero_f1(self) -> None:
        """一致が無ければ F1 は 0 とする。"""
        assert match_multiset(["x"], ["y"]).f1 == 0.0


class TestEvaluatePage:
    """1 ページの評価。"""

    def test_scores_prose_math_and_headings(self) -> None:
        """地の文・数式・見出しをそれぞれ比較する。"""
        truth = "## 4.3 累乗根\n\n無限に続く。$(\\sqrt[4]{a})^4 = a$"
        pred = "### 4.3 累乗根\n\n無制限に続く。$(\\sqrt[3]{a})^4 = a$"

        scores = evaluate_page(truth, pred)

        assert scores.cer == CerScore(edits=1, truth_length=len("4.3累乗根無限に続く。"))
        assert scores.math == MatchScore(matched=0, truth_count=1, pred_count=1)
        assert scores.headings == MatchScore(matched=0, truth_count=1, pred_count=1)

    def test_formula_notation_differences_are_not_errors(self) -> None:
        """表記揺れだけの数式は一致とみなす。"""
        scores = evaluate_page("$\\dfrac{1}{2}$", "$\\frac{1}{2}$")

        assert scores.math.matched == 1


class TestSummarize:
    """全ページの集計。"""

    def test_sums_counts_over_pages(self) -> None:
        """ページごとの件数を合算してから比率を計算する（マイクロ平均）。"""
        first = evaluate_page("あいうえお", "あいうえお")
        second = evaluate_page("かきくけこ$x$", "かきくけ$y$")

        total = summarize([first, second])

        assert total.cer == CerScore(edits=1, truth_length=10)
        assert total.math == MatchScore(matched=0, truth_count=1, pred_count=1)
