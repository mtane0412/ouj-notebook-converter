"""仕様: detectors.equation_check モジュールのユニットテスト。

数式の LaTeX から等式・不等式の連鎖を「左辺・関係記号・右辺」の組（リンク）に分解する extract_links、
リンクを記号計算で検算する check_link、1 ページ分の Markdown をまとめて検算する check_markdown をテストする。
"""

import time

import pytest

from ouj_notebook_converter.detectors import equation_check
from ouj_notebook_converter.detectors.equation_check import (
    Link,
    LinkStatus,
    check_link,
    check_markdown,
    extract_links,
)


def _triples(links: list[Link] | None) -> list[tuple[str, str, str]]:
    assert links is not None
    return [(link.lhs, link.relation, link.rhs) for link in links]


class TestExtractLinks:
    """extract_links: 数式を関係式のリンクに分解する。"""

    def test_splits_chain_into_adjacent_pairs(self) -> None:
        """等式と不等式の連鎖を、隣り合う 2 項ずつのリンクに分解する。"""
        assert _triples(extract_links("a = b > c")) == [("a", "=", "b"), ("b", ">", "c")]

    def test_does_not_split_at_relation_inside_braces(self) -> None:
        """総和の添字（{k=1}）のように括弧の中にある等号では分割しない。"""
        assert _triples(extract_links(r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}")) == [
            (r"\sum_{k=1}^{n} k", "=", r"\frac{n(n+1)}{2}")
        ]

    def test_does_not_chain_statements_separated_by_comma(self) -> None:
        """カンマで区切られた別々の式（x = 1 と y = 2）は連鎖としてつながない。"""
        assert _triples(extract_links("x = 1, y = 2")) == [("x", "=", "1"), ("y", "=", "2")]

    def test_text_separates_statements_and_is_recorded_as_following_text(self) -> None:
        """\\text{...} の日本語で文を区切り、その日本語を直前の文の直後の文として記録する。"""
        links = extract_links(r"a = b \text{ ではない。} c = d")
        assert _triples(links) == [("a", "=", "b"), ("c", "=", "d")]
        assert links is not None
        assert links[0].following_text == " ではない。"

    def test_text_at_end_of_formula_is_recorded_as_following_text(self) -> None:
        """数式の末尾の \\text{...} も、最後の文の直後の文として記録する。"""
        links = extract_links(r"\sqrt{(-2)^2} = -2 \text{ ではない。}", following_text="よって")
        assert links is not None
        assert links[-1].following_text == " ではない。よって"

    def test_last_statement_gets_prose_after_formula(self) -> None:
        """数式の最後の文には、数式の後ろの地の文を直後の文として記録する。"""
        links = extract_links("a = b", following_text="ではない。")
        assert links is not None
        assert links[0].following_text == "ではない。"

    def test_aligned_line_starting_with_relation_continues_chain(self) -> None:
        """aligned で行頭が関係記号の行（&= C）は、前の行の最後の項から連鎖を続ける。"""
        tex = r"\begin{aligned} A &= B \\ &= C \end{aligned}"
        assert _triples(extract_links(tex)) == [("A", "=", "B"), ("B", "=", "C")]

    def test_returns_none_for_unsupported_environments(self) -> None:
        """筆算（array）や場合分け（cases）の環境を含む数式は検算の対象外として None を返す。"""
        assert extract_links(r"\begin{array}{r} 12 \\ +34 \\ \hline 46 \end{array}") is None
        assert extract_links(r"|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}") is None

    def test_formula_without_relation_has_no_links(self) -> None:
        """関係記号を含まない数式はリンクを持たない。"""
        assert _triples(extract_links(r"\sqrt{2}")) == []


class TestCheckLink:
    """check_link: リンクを記号計算で検算する。"""

    @staticmethod
    def _check(lhs: str, relation: str, rhs: str, following_text: str = "") -> LinkStatus:
        return check_link(Link(lhs=lhs, relation=relation, rhs=rhs, following_text=following_text))

    def test_flags_p70_misread_of_fourth_root_as_cube_root(self) -> None:
        """p.70 で 4 乗根を 3 乗根と誤読した等式は恒等的に成り立たないため誤読候補にする。"""
        assert self._check(r"(\sqrt[3]{a})^3", "=", r"(\sqrt[3]{a})^4") == LinkStatus.VIOLATED

    def test_identity_holds(self) -> None:
        """恒等式は成り立つと判定する。"""
        assert self._check("(x+1)^2", "=", "x^2 + 2x + 1") == LinkStatus.HOLDS

    def test_flags_wrong_numeric_equality(self) -> None:
        """文字を含まない誤った等式は誤読候補にする。"""
        assert self._check("2 + 3", "=", "6") == LinkStatus.VIOLATED

    def test_equality_with_single_symbol_side_is_conditional(self) -> None:
        """片辺が文字 1 つの等式（y = 2x）は定義や解とみなし、誤読候補にしない。"""
        assert self._check("y", "=", "2x") == LinkStatus.CONDITIONAL

    def test_equality_with_number_side_is_conditional(self) -> None:
        """片辺が数値の等式（... = 0）は方程式とみなし、誤読候補にしない。"""
        assert self._check("x^2 - 3x + 2", "=", "0") == LinkStatus.CONDITIONAL

    def test_equality_with_different_symbol_sets_is_conditional(self) -> None:
        """両辺の文字の組が異なる等式（x^2 = y^3）は方程式とみなし、誤読候補にしない。"""
        assert self._check("x^2", "=", "y^3") == LinkStatus.CONDITIONAL

    def test_inequality_depending_on_values_is_conditional(self) -> None:
        """文字の値によって成否が変わる不等式（a^2 > b^2）は誤読候補にしない。"""
        assert self._check("a^2", ">", "b^2") == LinkStatus.CONDITIONAL

    def test_flags_inequality_false_for_all_values(self) -> None:
        """どの値でも成り立たない不等式（x^2 + 1 < 0）は誤読候補にする。"""
        assert self._check("x^2 + 1", "<", "0") == LinkStatus.VIOLATED

    def test_condition_on_single_symbol_is_conditional(self) -> None:
        """文字 1 つの範囲の条件（a \\le -3）は、負の大きな値でも試して誤読候補にしない。"""
        assert self._check("a", r"\le", "-3") == LinkStatus.CONDITIONAL
        assert self._check("x", ">", "100") == LinkStatus.CONDITIONAL

    def test_decimal_bounds_of_inequality_are_compared_exactly(self) -> None:
        """不等式の小数（1.4 < \\sqrt{2}）には近似の許容誤差を使わず、そのまま比較する。"""
        assert self._check("1.4", "<", r"\sqrt{2}") == LinkStatus.HOLDS
        assert self._check(r"\sqrt{2}", "<", "1.5") == LinkStatus.HOLDS

    def test_summation_upper_limit_takes_integer_values(self) -> None:
        """総和の上限 n には整数を代入して検算する。"""
        tex_sum = r"\frac{1}{2} \sum_{k=1}^n \left(\frac{1}{2}\right)^{k-1}"
        tex_closed = r"\frac{1}{2} \cdot \frac{1 - \left(\frac{1}{2}\right)^n}{1 - \frac{1}{2}}"
        assert self._check(tex_sum, "=", tex_closed) == LinkStatus.HOLDS
        assert self._check(r"\sum_{k=1}^n k", "=", r"\frac{n(n+1)}{2}") == LinkStatus.HOLDS

    def test_skips_fragment_split_by_line_break(self) -> None:
        """改行や \\cdots で切れて + で始まる式の断片は、検算せずスキップする。"""
        assert self._check(r"+ (\frac{1}{n-1} - \frac{1}{n})", "=", r"1 - \frac{1}{n}") == (
            LinkStatus.SKIPPED_PARSE
        )
        assert self._check(r"1 + 2 +", "=", "3") == LinkStatus.SKIPPED_PARSE

    def test_skips_derivative_with_prime(self) -> None:
        """f'(0) のようなプライム記号の微分は数値計算できないため、検算せずスキップする。"""
        assert self._check("1", "=", "f'(0)") == LinkStatus.SKIPPED_PARSE

    def test_i_as_imaginary_unit(self) -> None:
        """文字 i を虚数単位とみなすと成り立つ等式は、成り立つと判定する。"""
        assert self._check("(1+i)^2", "=", "2i") == LinkStatus.HOLDS

    def test_decimal_compared_with_its_written_precision(self) -> None:
        """小数の近似値は表記された桁数の精度で比較する。

        切り捨て（1.41）も四捨五入も許し、表記の最小桁より大きくずれていれば誤読候補にする。
        """
        assert self._check(r"\sqrt{2}", "=", "1.41") == LinkStatus.HOLDS
        assert self._check(r"\sqrt{2}", "=", "1.44") == LinkStatus.VIOLATED

    def test_ambiguous_function_application_or_product(self) -> None:
        """関数適用と積の 2 通りに読める式（x(x+1)）は、どちらかで成り立てば成り立つと判定する。"""
        assert self._check("x(x+1)", "=", "x^2 + x") == LinkStatus.HOLDS

    def test_pi(self) -> None:
        """円周率 \\pi を含む式を検算する。"""
        assert self._check(r"\sin \frac{\pi}{6}", "=", r"\frac{1}{2}") == LinkStatus.HOLDS
        assert self._check(r"\pi", "=", "3.14") == LinkStatus.HOLDS

    def test_degree_converted_to_radian(self) -> None:
        """度数法の角度（30^\\circ）を弧度法に直して検算する。"""
        assert self._check(r"\sin 30^\circ", "=", r"\frac{1}{2}") == LinkStatus.HOLDS

    def test_negated_by_following_text(self) -> None:
        """直後の文（「ではない」「としてはいけない」）で否定されている誤った式は誤読候補にしない。"""
        assert (
            self._check(r"\sqrt{4}", "=", "-2", following_text=" ではない。") == LinkStatus.NEGATED
        )
        assert (
            self._check(
                r"\frac{5+3c}{3d}", "=", r"\frac{5+c}{d}", following_text="としてはいけない"
            )
            == LinkStatus.NEGATED
        )

    def test_skips_term_with_japanese(self) -> None:
        """日本語を含む項は検算せずスキップする。"""
        assert self._check("x", "=", "2 \\text{倍}") == LinkStatus.SKIPPED_NON_ASCII
        assert self._check("面積", "=", "ab") == LinkStatus.SKIPPED_NON_ASCII

    def test_skips_empty_term(self) -> None:
        """空の項（数式が関係記号で始まる場合など）は検算せずスキップする。"""
        assert self._check("", "=", "a") == LinkStatus.SKIPPED_EMPTY

    def test_skips_unparsable_term(self) -> None:
        """LaTeX の式として解釈できない項（\\cdots を含む和など）は検算せずスキップする。"""
        assert self._check(r"1 + 2 + \cdots + n", "=", r"\frac{n(n+1)}{2}") == (
            LinkStatus.SKIPPED_PARSE
        )

    def test_skips_link_exceeding_time_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """数値計算が時間制限を超えたリンク（総和の上限に小数を代入した場合など）は打ち切ってスキップする。"""

        def slow_evaluation(*_args: object) -> list[LinkStatus]:
            time.sleep(1)
            return [LinkStatus.HOLDS]

        monkeypatch.setattr(equation_check, "LINK_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(equation_check, "_evaluate_variants", slow_evaluation)
        assert self._check("a + b", "=", "b + a") == LinkStatus.SKIPPED_TIMEOUT

    @pytest.mark.parametrize("relation", [r"\le", r"\leq", r"\leqq", r"\ge", r"\geq", r"\geqq"])
    def test_all_aliases_of_non_strict_inequality(self, relation: str) -> None:
        """等号付き不等号の別名をすべて検算できる。"""
        assert self._check("2", relation, "2") == LinkStatus.HOLDS

    def test_not_equal(self) -> None:
        """\\neq は値が異なれば成り立ち、等しければ誤読候補にする。"""
        assert self._check("1", r"\neq", "2") == LinkStatus.HOLDS
        assert self._check("2", r"\ne", "2") == LinkStatus.VIOLATED


class TestCheckMarkdown:
    """check_markdown: 1 ページ分の Markdown の全数式を検算する。"""

    def test_returns_violations_with_formula_index(self) -> None:
        """ページ内の誤読候補を、数式の出現順の番号とともに返す。"""
        markdown = (
            "$a > 1$ のとき，\n\n"
            "$$(\\sqrt[2]{a})^2 = a = (\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4 \\text{となり，}$$\n"
        )
        result = check_markdown(markdown)
        violated = [r for r in result.links if r.status == LinkStatus.VIOLATED]
        assert len(violated) == 1
        assert violated[0].formula_index == 1
        assert violated[0].link.rhs == r"(\sqrt[3]{a})^4"
        assert result.formula_count == 2
        assert result.skipped_formula_count == 0

    def test_negated_by_prose_after_formula(self) -> None:
        """数式の後ろの地の文で否定されていれば、誤読候補にしない。"""
        result = check_markdown("つまり $\\sqrt{4} = -2$ ではない。")
        assert [r.status for r in result.links] == [LinkStatus.NEGATED]

    def test_counts_formulas_with_unsupported_environment(self) -> None:
        """検算の対象外の環境（筆算の array）を含む数式の件数を数える。"""
        result = check_markdown("$$\\begin{array}{r} 12 \\\\ +34 \\end{array}$$\n\n$x = 1$")
        assert result.skipped_formula_count == 1
        assert len(result.links) == 1
