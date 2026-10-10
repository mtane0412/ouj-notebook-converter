"""仕様: detectors.multi_run_variance.voting モジュールのユニットテスト。

複数回の OCR 結果を多数決で確定し、単独の実行より評価指標が改善することをテストする。
"""

from ouj_notebook_converter.detectors.multi_run_variance.voting import (
    parts_of,
    pick_medoid,
    score_parts,
    vote_page,
    vote_sequence,
)
from ouj_notebook_converter.evaluation.markdown_parts import split_markdown


class TestVoteSequence:
    """vote_sequence: 食い違い箇所ごとに最も多い版を採用する。"""

    def test_majority_variant_wins(self) -> None:
        voted = vote_sequence(["著者は服部です", "著者は隈部です", "著者は隈部です"])
        assert voted == "著者は隈部です"

    def test_tie_prefers_reference(self) -> None:
        assert vote_sequence(["著者は服部です", "著者は隈部です"], reference=0) == "著者は服部です"

    def test_three_way_tie_prefers_reference(self) -> None:
        assert vote_sequence(["あAう", "あBう", "あCう"], reference=1) == "あBう"

    def test_independent_errors_in_different_runs_are_all_corrected(self) -> None:
        voted = vote_sequence(
            ["あXうえおかきくけこ", "あいうえおかYくけこ", "あいうえおかきくけこ"]
        )
        assert voted == "あいうえおかきくけこ"

    def test_votes_sequences_of_formulas(self) -> None:
        voted = vote_sequence([("x^3", "y"), ("x^2", "y"), ("x^2", "y")])
        assert voted == ("x^2", "y")


class TestPickMedoid:
    def test_run_closest_to_the_others_is_chosen(self) -> None:
        assert pick_medoid(["著者は服部です", "著者は隈部です", "著者は隈部です"]) == 1

    def test_tie_prefers_earlier_run(self) -> None:
        assert pick_medoid(["あいう", "あいう"]) == 0


class TestPartsOf:
    def test_prose_matches_evaluation_prose(self) -> None:
        markdown = "## 見出し\n\n有理数 $a+b$ の和は\\(\\text{整数}\\)です。\n"
        assert parts_of(markdown).prose == split_markdown(markdown).prose

    def test_formulas_are_normalized_and_empty_ones_dropped(self) -> None:
        parts = parts_of("$\\dfrac{1}{2}$ と $\\text{ただし}$")
        assert parts.formulas == ("\\frac{1}{2}",)


class TestVotePage:
    def test_vote_corrects_prose_and_formula_errors(self) -> None:
        truth = "著者は隈部です。ここで $x^2+1$ を得る。"
        runs = [
            "著者は服部です。ここで $x^2+1$ を得る。",
            "著者は隈部です。ここで $x^3+1$ を得る。",
            "著者は隈部です。ここで $x^2+1$ を得る。",
        ]
        assert vote_page(runs) == parts_of(truth)

    def test_voted_score_is_better_than_single_run(self) -> None:
        truth = parts_of("著者は隈部です。ここで $x^2+1$ を得る。")
        runs = [
            "著者は服部です。ここで $x^2+1$ を得る。",
            "著者は隈部です。ここで $x^3+1$ を得る。",
            "著者は隈部です。ここで $x^2+1$ を得る。",
        ]
        voted = score_parts(truth, vote_page(runs))
        assert score_parts(truth, parts_of(runs[0])).cer.edits == 1
        assert voted.cer.edits == 0
        assert voted.math.f1 == 1.0
        assert score_parts(truth, parts_of(runs[1])).math.f1 < 1.0
