"""仕様: evaluation.markdown_parts モジュール（評価用の Markdown 分解）のユニットテスト。

1 ページ分の Markdown を「地の文（日本語部分）」「数式」「見出し」に分解する
split_markdown と、数式比較用の LaTeX 正規化 normalize_latex をテストする。
"""

from ouj_notebook_converter.evaluation.markdown_parts import (
    Formula,
    Heading,
    normalize_latex,
    split_markdown,
)


class TestProse:
    """地の文（CER の比較対象）の抽出。"""

    def test_removes_math_and_whitespace(self) -> None:
        """数式と空白を取り除いた日本語部分だけを残す。"""
        parts = split_markdown("ここで $x = 1$ とおくと，\n\n$$y = 2$$\n\nとなる。")

        assert parts.prose == "ここでとおくと,となる。"

    def test_removes_markdown_markup(self) -> None:
        """見出し記号・強調・表の罫線・リスト記号・画像を取り除く。"""
        markdown = (
            "## 1.2 平方根\n"
            "**例 1.3** 次を求めよ。\n"
            "- 一つ目\n"
            "| 項目 | 値 |\n"
            "| --- | --- |\n"
            "| 長さ | 三 |\n"
            "![図 1.1](figures/fig.png)\n"
        )

        assert split_markdown(markdown).prose == "1.2平方根例1.3次を求めよ。一つ目項目値長さ三"

    def test_unifies_full_and_half_width_characters(self) -> None:
        """全角英数字と半角英数字の違いを誤りとして数えないよう NFKC で正規化する。"""
        assert split_markdown("第１章　ＡＢＣ").prose == "第1章ABC"


class TestFormulas:
    """数式の抽出。"""

    def test_extracts_inline_and_display_formulas_in_order(self) -> None:
        """インライン数式とディスプレイ数式を出現順に取り出す。"""
        parts = split_markdown(
            "まず $a$ を考える。\n\n$$b + c$$\n\n次に \\(d\\) と \\[e\\] を見る。"
        )

        assert parts.formulas == (
            Formula(tex="a", display=False),
            Formula(tex="b + c", display=True),
            Formula(tex="d", display=False),
            Formula(tex="e", display=True),
        )

    def test_keeps_environment_as_display_formula(self) -> None:
        """数式外に置かれた LaTeX 環境は環境ごと 1 つのディスプレイ数式として扱う。"""
        markdown = "\\begin{align}a &= b\\end{align}"

        assert split_markdown(markdown).formulas == (Formula(tex=markdown, display=True),)


class TestHeadings:
    """見出しの抽出。"""

    def test_extracts_level_and_text(self) -> None:
        """見出しのレベルと、強調記号・空白を除いたテキストを取り出す。"""
        parts = split_markdown("# 第 1 章　数と式\n\n本文\n\n## **1.1 自然数**\n")

        assert parts.headings == (
            Heading(level=1, text="第1章数と式"),
            Heading(level=2, text="1.1自然数"),
        )

    def test_ignores_hash_inside_code_fence(self) -> None:
        """コードフェンス内の # 行は見出しとして扱わない。"""
        assert split_markdown("```\n# 図中の文字\n```\n").headings == ()


class TestNormalizeLatex:
    """数式比較用の LaTeX 正規化。"""

    def test_ignores_whitespace_and_spacing_commands(self) -> None:
        """空白と間隔調整命令の違いを無視する。"""
        assert normalize_latex("a \\, + \\quad b") == normalize_latex("a+b")

    def test_unifies_equivalent_commands(self) -> None:
        """表示上同じになる命令の表記揺れを統一する。"""
        assert normalize_latex("\\dfrac{1}{2} \\le \\left( x \\right)") == normalize_latex(
            "\\frac{1}{2}\\leq(x)"
        )

    def test_ignores_trailing_punctuation(self) -> None:
        """数式末尾の句読点の有無を無視する。"""
        assert normalize_latex("x = 1,") == normalize_latex("x = 1")

    def test_keeps_different_root_index(self) -> None:
        """根指数の違い（実際に起きた 4 乗根→3 乗根の誤読）は区別する。"""
        assert normalize_latex("\\sqrt[4]{a}") != normalize_latex("\\sqrt[3]{a}")

    def test_does_not_break_longer_command_names(self) -> None:
        """\\le の置換が \\left や \\leftarrow など別の命令を壊さない。"""
        assert normalize_latex("\\leftarrow") == "\\leftarrow"
