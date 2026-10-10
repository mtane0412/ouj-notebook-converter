"""仕様: detectors.multi_run_variance.detect モジュールのユニットテスト。

同一ページの複数回の OCR 結果を整列し、実行間で食い違う箇所（地の文・数式）を
不確実箇所の候補として抽出することをテストする。
"""

from ouj_notebook_converter.detectors.multi_run_variance.detect import (
    DetectOptions,
    align_runs,
    detect_page,
)


class TestAlignRuns:
    """align_runs: 基準の実行に対し、他の実行と食い違う範囲を実行ごとの範囲つきでまとめる。"""

    def test_identical_runs_have_no_clusters(self) -> None:
        assert align_runs(["有理数の性質", "有理数の性質", "有理数の性質"]) == []

    def test_replacement_in_one_run_gives_one_cluster_with_each_run_span(self) -> None:
        clusters = align_runs(["著者は隈部です", "著者は服部です", "著者は隈部です"])
        assert len(clusters) == 1
        # 基準（0 番）では「隈」(3, 4)、1 番では「服」(3, 4)
        assert clusters[0].spans == ((3, 4), (3, 4), (3, 4))

    def test_overlapping_differences_from_different_runs_are_merged(self) -> None:
        clusters = align_runs(["あいうえお", "あXXえお", "あいYYお"])
        assert len(clusters) == 1
        # 1 番は「いう」→「XX」、2 番は「うえ」→「YY」。重なるので基準では「いうえ」(1, 4) にまとまる
        assert clusters[0].spans == ((1, 4), (1, 4), (1, 4))

    def test_missing_character_in_other_run_has_shorter_span(self) -> None:
        clusters = align_runs(["微分積分学", "微積分学"])
        assert len(clusters) == 1
        # 基準の「分積」は 1 番では「積」。difflib は重複した「分」の片方を削除とみなす
        assert clusters[0].spans == ((1, 2), (1, 1))

    def test_insertion_relative_to_reference_has_empty_reference_span(self) -> None:
        clusters = align_runs(["微積分学", "微分積分学"])
        start, end = clusters[0].spans[0]
        assert start == end

    def test_reference_can_be_chosen(self) -> None:
        clusters = align_runs(["著者は隈部です", "著者は服部です"], reference=1)
        assert clusters[0].spans == ((3, 4), (3, 4))

    def test_works_on_sequences_of_formulas(self) -> None:
        clusters = align_runs([("x^2", "y"), ("x^3", "y")])
        assert clusters[0].spans == ((0, 1), (0, 1))


class TestDetectPageProse:
    """detect_page: 地の文の食い違い。"""

    def test_reports_text_of_every_run_with_context(self) -> None:
        runs = ["著者は隈部正博です。", "著者は服部正博です。", "著者は隈部正博です。"]
        candidates = detect_page(3, runs)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.page == 3
        assert c.granularity == "prose"
        assert c.texts == ("隈", "服", "隈")
        assert c.context_before == "著者は"
        assert c.context_after == "部正博です。"
        assert c.dissent == 1
        assert c.run_count == 3

    def test_dissent_counts_runs_that_differ_from_reference(self) -> None:
        runs = ["著者は隈部です", "著者は服部です", "著者は猥部です"]
        assert detect_page(3, runs)[0].dissent == 2

    def test_min_dissent_filters_candidates(self) -> None:
        runs = ["著者は隈部です", "著者は服部です", "著者は隈部です"]
        assert detect_page(3, runs, options=DetectOptions(min_dissent=2)) == []

    def test_identical_runs_have_no_candidates(self) -> None:
        assert detect_page(3, ["有理数と無理数"] * 3) == []

    def test_markdown_markup_differences_are_not_candidates(self) -> None:
        runs = ["## 有理数\n\n**定義**です", "有理数\n\n定義です"]
        assert detect_page(3, runs) == []

    def test_difference_only_in_number_of_formulas_is_not_a_prose_candidate(self) -> None:
        runs = ["式 $x$ を考える", "式を考える"]
        assert [c for c in detect_page(3, runs) if c.granularity == "prose"] == []

    def test_math_position_in_context_is_labeled(self) -> None:
        runs = ["式 $x$ の値は三です", "式 $x$ の値は二です"]
        c = detect_page(3, runs)[0]
        assert c.context_before.endswith("〔数式〕の値は")


class TestDetectPageFormula:
    """detect_page: 数式の食い違い。"""

    def test_different_formula_is_a_candidate_with_reference_index(self) -> None:
        runs = [
            "最初に $a+b$ を、次に $x^2$ を考える",
            "最初に $a+b$ を、次に $x^3$ を考える",
        ]
        formula = [c for c in detect_page(5, runs) if c.granularity == "formula"]
        assert len(formula) == 1
        assert formula[0].texts == ("x^2", "x^3")
        assert formula[0].formula_indices == (1,)
        assert formula[0].dissent == 1

    def test_notation_difference_is_not_a_candidate(self) -> None:
        """\\dfrac と \\frac、空白の違いは表示が同じなので食い違いにしない。"""
        runs = ["$\\dfrac{1}{2}$ です", "$\\frac{1}{2}$ です"]
        assert [c for c in detect_page(5, runs) if c.granularity == "formula"] == []

    def test_extra_formula_in_other_run_has_no_reference_index(self) -> None:
        runs = ["$a$ と $b$", "$a$ と $b$ と $c$"]
        formula = [c for c in detect_page(5, runs) if c.granularity == "formula"]
        assert len(formula) == 1
        assert formula[0].formula_indices == ()
        assert formula[0].texts == ("", "c")

    def test_to_dict_has_page_runs_texts_and_context(self) -> None:
        runs = ["著者は隈部です", "著者は服部です"]
        data = detect_page(3, runs)[0].to_dict()
        assert data["page"] == 3
        assert data["granularity"] == "prose"
        assert data["texts"] == ["隈", "服"]
        assert data["context_before"] == "著者は"
        assert data["context_after"] == "部です"
        assert data["reference_run"] == 0
