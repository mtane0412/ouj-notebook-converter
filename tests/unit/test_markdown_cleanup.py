"""仕様: pipeline.stages.markdown_cleanup モジュール（OCR Markdown の後処理）のユニットテスト。

OCR（主に Gemini）が出力した Markdown に残る LaTeX の残骸と見出しレベルの揺れを
正規化する normalize_ocr_markdown をテストする。
テストデータは「初歩からの数学」の OCR 結果で実際に見つかった破綻パターンを元にしている。
"""

from ouj_notebook_converter.pipeline.stages.markdown_cleanup import normalize_ocr_markdown


class TestHfill:
    """数式外に残った \\hfill の除去。"""

    def test_removes_hfill_in_prose_and_keeps_equation_number(self) -> None:
        """地の文のhfillを除去して式番号を残す。"""
        text = "マイナスの符号を付けることは，$-1$ 倍することに等しい。 \\hfill (1.18)"

        assert normalize_ocr_markdown(text) == (
            "マイナスの符号を付けることは，$-1$ 倍することに等しい。 (1.18)"
        )

    def test_keeps_hfill_inside_math(self) -> None:
        """数式内のhfillは変更しない。"""
        text = "$$a = b \\hfill (1.1)$$"

        assert normalize_ocr_markdown(text) == text


class TestTag:
    """\\tag{...} を式番号のテキストへ変換する。"""

    def test_converts_tag_in_prose_to_parenthesized_number(self) -> None:
        """地の文のtagを括弧付き式番号にする。"""
        text = "つまり，$\\lim_{x \\to c} f(x) = f(c)$ が成り立つ。 \\tag{13.11}"

        assert normalize_ocr_markdown(text) == (
            "つまり，$\\lim_{x \\to c} f(x) = f(c)$ が成り立つ。 (13.11)"
        )

    def test_converts_tag_only_inline_math_to_number(self) -> None:
        """tagだけのインライン数式を式番号にする。"""
        text = "$a + c = b$ ならば $a = b - c$ $\\tag{2.11}$"

        assert normalize_ocr_markdown(text) == "$a + c = b$ ならば $a = b - c$ (2.11)"

    def test_converts_tag_only_display_math_to_number(self) -> None:
        """tagだけのディスプレイ数式を式番号にする。"""
        text = "$b = 0$ か少なくとも一方が成り立つ。\n$$ \\tag{1.10} $$\n\n次の段落"

        assert normalize_ocr_markdown(text) == (
            "$b = 0$ か少なくとも一方が成り立つ。\n(1.10)\n\n次の段落"
        )

    def test_converts_tag_only_equation_environment_to_number(self) -> None:
        """tagだけのequation環境を式番号にする。"""
        text = "弧長に比例する。  \n\\begin{equation}\n\\tag{6.1}\n\\end{equation}\n"

        assert normalize_ocr_markdown(text) == "弧長に比例する。  \n(6.1)\n"

    def test_keeps_tag_in_math_with_body(self) -> None:
        """式本体を持つ数式のtagは変更しない。"""
        text = "$$x^2 = 1 \\tag{5.1}$$"

        assert normalize_ocr_markdown(text) == text

    def test_keeps_tag_only_non_equation_environment(self) -> None:
        """equation 以外の環境（eqnarray* など）は tag だけでも変更しない。"""
        text = "\\begin{eqnarray*}\n\\tag{1.4}\n\\end{eqnarray*}"

        assert normalize_ocr_markdown(text) == text


class TestQuad:
    """数式外に残った \\quad / \\qquad を全角スペースに置き換える。"""

    def test_replaces_quad_in_prose_with_fullwidth_spaces(self) -> None:
        """地の文のquadを全角スペースにする。"""
        text = "**解答** (i) $-5$ \\quad (ii) $-1$ \\qquad (iii) $0$"

        assert normalize_ocr_markdown(text) == "**解答** (i) $-5$ 　 (ii) $-1$ 　　 (iii) $0$"

    def test_keeps_quad_inside_math(self) -> None:
        """数式内のquadは変更しない。"""
        text = "$$a = b \\quad (1.2)$$ と $x \\qquad y$"

        assert normalize_ocr_markdown(text) == text


class TestSectionHeading:
    """節番号（N.M）を持つ見出しのレベルを ## に揃える。"""

    def test_promotes_h3_section_heading_to_h2(self) -> None:
        """h3の節見出しをh2にする。"""
        assert normalize_ocr_markdown("### 13.5 無限数列の和 (A)") == "## 13.5 無限数列の和 (A)"

    def test_keeps_h2_section_heading(self) -> None:
        """h2の節見出しは変更しない。"""
        text = "## 4.4 $n$ 乗根 (A)"

        assert normalize_ocr_markdown(text) == text

    def test_keeps_heading_with_three_level_number(self) -> None:
        """3階層の番号を持つ見出しは変更しない。"""
        text = "### 1.2.3 小節の見出し"

        assert normalize_ocr_markdown(text) == text

    def test_keeps_chapter_heading(self) -> None:
        """章見出しは変更しない。"""
        text = "# 6 図形の性質"

        assert normalize_ocr_markdown(text) == text


class TestLabelHeading:
    """例・コメント・練習などの見出し化されたラベルを太字に戻す。"""

    def test_converts_label_only_heading_to_bold(self) -> None:
        """ラベルだけの見出しを太字にする。"""
        assert normalize_ocr_markdown("### 例 4.4") == "**例 4.4**"

    def test_bolds_only_label_when_heading_has_trailing_text(self) -> None:
        """後続の文がある見出しはラベルだけを太字にする。"""
        text = "### 例 15.8　例 15.1, 例 15.2, 例 15.3 を思い出そう。"

        assert normalize_ocr_markdown(text) == (
            "**例 15.8**　例 15.1, 例 15.2, 例 15.3 を思い出そう。"
        )

    def test_includes_difficulty_mark_in_label(self) -> None:
        """難易度記号もラベルに含める。"""
        text = "### コメント 1.3 (C)　(1.21) をより詳細に理解しよう。"

        assert normalize_ocr_markdown(text) == (
            "**コメント 1.3 (C)**　(1.21) をより詳細に理解しよう。"
        )

    def test_separates_trailing_text_with_fullwidth_space(self) -> None:
        """半角スペース区切りの後続文は全角スペースで区切る。"""
        assert normalize_ocr_markdown("### 練習 5.1 次の計算をせよ。") == (
            "**練習 5.1**　次の計算をせよ。"
        )

    def test_converts_h4_label_heading_to_bold(self) -> None:
        """h4の見出しも太字にする。"""
        assert normalize_ocr_markdown("#### 例 13.11") == "**例 13.11**"

    def test_converts_theorem_heading_to_bold(self) -> None:
        """定理の見出しも太字にする。"""
        assert normalize_ocr_markdown("### 定理 5.2") == "**定理 5.2**"

    def test_keeps_heading_without_number(self) -> None:
        """番号の無い見出しは変更しない。"""
        text = "### 例題について"

        assert normalize_ocr_markdown(text) == text


class TestCodeFence:
    """コードフェンス内（図のテキスト表現）は変更しない。"""

    def test_keeps_code_fence_content(self) -> None:
        """コードフェンス内は変更しない。"""
        text = "```\n### 例 1.1\n\\hfill \\tag{1.1}\n```"

        assert normalize_ocr_markdown(text) == text


class TestNoChange:
    """破綻の無い Markdown はそのまま返す。"""

    def test_keeps_markdown_without_breakage(self) -> None:
        """通常の本文は変更しない。"""
        text = "# 1 数の概念\n\n## 1.1 自然数 (A)\n\n**例 1.1** 自然数の和 $1 + 2 = 3$ を考える。\n"

        assert normalize_ocr_markdown(text) == text
