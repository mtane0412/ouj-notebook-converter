"""仕様: exporters.text_layer モジュール（searchable PDF のテキストレイヤー作成）のユニットテスト。

- flatten_latex: 数式を検索しやすい平文に崩す
- markdown_to_search_text: Gemini の Markdown をテキストレイヤー用の文字列にする
- load_ordered_words: yomitoku の analysis.json の単語を読み順に並べる
- assign_text: yomitoku の単語ごとに、Gemini の文字列のうち対応する部分を割り当てる
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ouj_notebook_converter.exporters.text_layer import (
    AlignmentStats,
    assign_text,
    flatten_latex,
    load_ordered_words,
    markdown_to_search_text,
)


class TestFlattenLatex:
    """数式の平文化。"""

    def test_removes_delimiters_and_scripts(self) -> None:
        """区切り記号・上付き・下付き・波括弧を取り除く（x^2 は x2 になる）。"""
        assert flatten_latex("$x^{2} + a_1$") == "x2+a1"

    def test_converts_symbols_to_unicode(self) -> None:
        """記号の命令は対応する Unicode 文字にする。"""
        assert flatten_latex(r"$$a \times b \leq \pi$$") == "a×b≤π"

    def test_keeps_text_contents(self) -> None:
        """\\text{...} の中身は残す。"""
        assert flatten_latex(r"$0 + 1 \text{ の値は } 1$") == "0+1の値は1"

    def test_fraction_becomes_slash(self) -> None:
        """分数は「分子/分母」にする。"""
        assert flatten_latex(r"$\frac{1}{2} + \dfrac{a+b}{c}$") == "1/2+a+b/c"

    def test_tag_becomes_parenthesized_number(self) -> None:
        """式番号 \\tag{X} は (X) にする。"""
        assert flatten_latex(r"$$y = 1 \tag{1.1}$$") == "y=1(1.1)"

    def test_drops_environment_names_and_column_spec(self) -> None:
        """環境名と表の列指定・行区切り・列区切りは文字として残さない。"""
        tex = r"\begin{array}{rcl} 1 & + & 2 \\ \hline 3 & = & 3 \end{array}"

        assert flatten_latex(tex) == "1+23=3"

    def test_drops_layout_command_arguments(self) -> None:
        """\\hspace{2em} などのレイアウト命令の引数は文字として残さない。"""
        assert flatten_latex(r"$a \hspace{2em} b \enclose{longdiv}{12}$") == "ab12"

    def test_left_right_delimiters(self) -> None:
        """\\left( は ( にし、見えない区切り \\left. は何も残さない。"""
        assert flatten_latex(r"$\left( x \right) \left. y \right|$") == "(x)y|"

    def test_escaped_characters(self) -> None:
        """\\{ や \\% は記号として残し、間隔の命令は取り除く。"""
        assert flatten_latex(r"$\{ 1 \} \, 50\%$") == "{1}50%"

    def test_unknown_command_is_dropped(self) -> None:
        """対応表に無い命令は名前を残さず、引数の中身だけを残す。"""
        assert flatten_latex(r"$\overline{AB}$") == "AB"


class TestMarkdownToSearchText:
    """Markdown からテキストレイヤー用の文字列を作る。"""

    def test_flattens_math_and_removes_markup(self) -> None:
        """見出し記号・強調・空白を除き、数式は平文に崩して地の文に含める。"""
        markdown = "## 1.1 自然数\n\n**例 1.1** ここで $x^2 = 4$ とおく。\n"

        assert markdown_to_search_text(markdown) == "1.1自然数例1.1ここでx2=4とおく。"


def _word(
    content: str, x1: int, y1: int, x2: int, y2: int, direction: str = "horizontal"
) -> dict[str, Any]:
    return {
        "content": content,
        "direction": direction,
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadOrderedWords:
    """yomitoku の単語を読み順に並べる。"""

    def test_orders_by_paragraph_order_then_top_to_bottom(self, tmp_path: Path) -> None:
        """段落の order 順に、段落内は上の行から並べる（JSON の並びは下の行から）。"""
        data = {
            "paragraphs": [
                {"box": [0, 200, 500, 300], "order": 1, "direction": "horizontal"},
                {"box": [0, 0, 500, 100], "order": 0, "direction": "horizontal"},
            ],
            "figures": [],
            "words": [
                _word("二段落目の二行目", 0, 250, 400, 290),
                _word("二段落目の一行目", 0, 200, 400, 240),
                _word("一段落目", 0, 10, 400, 50),
            ],
        }
        path = _write_json(tmp_path / "analysis.json", data)

        words = load_ordered_words(path)

        assert [w.text for w in words] == ["一段落目", "二段落目の一行目", "二段落目の二行目"]

    def test_figure_paragraphs_follow_figure_order(self, tmp_path: Path) -> None:
        """図の中の段落は、図の order の位置に、図の中の order 順で並べる。"""
        data = {
            "paragraphs": [
                {"box": [0, 0, 500, 100], "order": 0, "direction": "horizontal"},
                {"box": [0, 400, 500, 500], "order": 2, "direction": "horizontal"},
            ],
            "figures": [
                {
                    "box": [0, 150, 500, 350],
                    "order": 1,
                    "paragraphs": [
                        {"box": [0, 300, 500, 350], "order": 1, "direction": "horizontal"},
                        {"box": [0, 150, 500, 200], "order": 0, "direction": "horizontal"},
                    ],
                }
            ],
            "words": [
                _word("本文の最後", 0, 410, 400, 450),
                _word("図の下の文字", 0, 310, 400, 340),
                _word("図の上の文字", 0, 160, 400, 190),
                _word("本文の最初", 0, 10, 400, 50),
            ],
        }
        path = _write_json(tmp_path / "analysis.json", data)

        words = load_ordered_words(path)

        assert [w.text for w in words] == [
            "本文の最初",
            "図の上の文字",
            "図の下の文字",
            "本文の最後",
        ]

    def test_words_outside_paragraphs_come_last_top_to_bottom(self, tmp_path: Path) -> None:
        """どの段落にも入らない単語（柱・ノンブルなど）は最後に、上から順に並べる。"""
        data = {
            "paragraphs": [{"box": [0, 100, 500, 200], "order": 0, "direction": "horizontal"}],
            "figures": [],
            "words": [
                _word("ノンブル", 0, 900, 50, 950),
                _word("本文", 0, 110, 400, 150),
                _word("柱", 0, 10, 50, 50),
            ],
        }
        path = _write_json(tmp_path / "analysis.json", data)

        words = load_ordered_words(path)

        assert [w.text for w in words] == ["本文", "柱", "ノンブル"]

    def test_vertical_paragraph_reads_right_to_left(self, tmp_path: Path) -> None:
        """縦書きの段落の中は右の列から並べる。"""
        data = {
            "paragraphs": [{"box": [0, 0, 500, 500], "order": 0, "direction": "vertical"}],
            "figures": [],
            "words": [
                _word("左の列", 0, 0, 50, 400, "vertical"),
                _word("右の列", 100, 0, 150, 400, "vertical"),
            ],
        }
        path = _write_json(tmp_path / "analysis.json", data)

        assert [w.text for w in load_ordered_words(path)] == ["右の列", "左の列"]

    def test_skips_words_without_content_or_points(self, tmp_path: Path) -> None:
        """文字か座標の無い単語は描画できないため除く。"""
        data = {
            "paragraphs": [],
            "figures": [],
            "words": [
                {"content": "", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {"content": "座標なし", "points": []},
                _word("有効", 0, 0, 10, 10),
            ],
        }
        path = _write_json(tmp_path / "analysis.json", data)

        words = load_ordered_words(path)

        assert [w.text for w in words] == ["有効"]
        assert words[0].bbox == (0, 0, 10, 10)


class TestAssignText:
    """yomitoku の単語への Gemini の文字の割り当て。"""

    def test_replaces_misread_characters_word_by_word(self) -> None:
        """一致する部分を足がかりに、各単語へ Gemini 側の対応部分を割り当てる。"""
        result = assign_text(["数の慨念", "について"], "数の概念について")

        assert result.texts == ("数の概念", "について")
        assert result.stats == AlignmentStats(
            words=2,
            words_without_text=0,
            chars=8,
            chars_matched=7,
            chars_replaced=1,
            chars_inserted=0,
        )

    def test_word_without_counterpart_becomes_empty(self) -> None:
        """Gemini に対応する文字が無い単語（柱など）は空にし、件数を数える。"""
        result = assign_text(["第1章数の概念11", "本文です"], "本文です")

        assert result.texts == ("", "本文です")
        assert result.stats.words_without_text == 1

    def test_inserted_characters_go_to_the_following_word(self) -> None:
        """yomitoku が読み落とした文字は、直後の単語に入れる（文字は捨てない）。"""
        result = assign_text(["自然数", "整数"], "自然数と整数")

        assert result.texts == ("自然数", "と整数")
        assert result.stats.chars_inserted == 1

    def test_trailing_and_leading_insertions_stay_on_page(self) -> None:
        """先頭・末尾にしか対応しない文字は、最初・最後の単語に入れる。"""
        result = assign_text(["本文"], "見出し本文脚注")

        assert result.texts == ("見出し本文脚注",)

    def test_all_characters_are_assigned_exactly_once(self) -> None:
        """割り当てた文字を順につなぐと、Gemini の文字列と一致する。"""
        text = "あいうえおかきくけこさしすせそ"
        result = assign_text(["かきく", "あいう", "けこ", "せそ"], text)

        assert "".join(result.texts) == text

    def test_normalizes_width_and_whitespace_before_aligning(self) -> None:
        """yomitoku の単語は NFKC 正規化・空白除去してから比べる。"""
        result = assign_text(["1+1の値は 2"], "1+1の値は2")

        assert result.stats.chars_matched == 7

    def test_no_words_with_text_raises(self) -> None:
        """単語が無いのに割り当てる文字がある場合は、文字を捨てずに停止する。"""
        with pytest.raises(ValueError, match="単語"):
            assign_text([], "本文")

    def test_stats_can_be_summed(self) -> None:
        """ページごとの件数を合算できる。"""
        a = AlignmentStats(
            words=1,
            words_without_text=0,
            chars=2,
            chars_matched=2,
            chars_replaced=0,
            chars_inserted=0,
        )
        b = AlignmentStats(
            words=2,
            words_without_text=1,
            chars=3,
            chars_matched=1,
            chars_replaced=1,
            chars_inserted=1,
        )

        assert a + b == AlignmentStats(
            words=3,
            words_without_text=1,
            chars=5,
            chars_matched=3,
            chars_replaced=1,
            chars_inserted=1,
        )
