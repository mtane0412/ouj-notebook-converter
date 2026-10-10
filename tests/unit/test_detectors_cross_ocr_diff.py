"""仕様: detectors.cross_ocr_diff.detect モジュールのユニットテスト。

Gemini の Markdown と yomitoku の analysis.json を日本語部分だけで比較し、
食い違い箇所を誤読候補として抽出できることを確かめる。
"""

from typing import Any

import pytest

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    MATH_MARKER,
    DiffOptions,
    build_comparable,
    compare_page,
    extract_gemini_prose,
    parse_analysis,
)


def _analysis(
    paragraphs: list[tuple[str, tuple[int, int, int, int]]],
    figures: list[tuple[int, int, int, int]] | None = None,
    words: list[tuple[str, tuple[int, int, int, int]]] | None = None,
) -> dict[str, Any]:
    """テスト用の yomitoku analysis.json（辞書）を作る。"""
    return {
        "paragraphs": [
            {"box": list(box), "contents": text, "role": None, "order": i}
            for i, (text, box) in enumerate(paragraphs)
        ],
        "figures": [{"box": list(box), "paragraphs": []} for box in (figures or [])],
        "tables": [],
        "words": [
            {
                "content": text,
                "points": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
            }
            for text, b in (words or [])
        ],
    }


class TestBuildComparable:
    """build_comparable: 日本語部分だけの比較用文字列と元の位置の対応を作る。"""

    def test_math_and_digits_become_separator(self) -> None:
        """日本語以外の連続（数字・記号・英字）は区切り文字 1 つにまとめる。"""
        text, index_map = build_comparable("0から6までの数")
        assert text == "\0から\0までの数"
        # 区切りは元の文字列で最初に現れた位置に対応する
        assert index_map[0] == 0
        assert index_map[1] == 1

    def test_punctuation_is_dropped_so_runs_continue(self) -> None:
        """句読点・括弧は取り除き、前後の日本語を同じ連続として扱う。"""
        text, _ = build_comparable("あるとき、は全単射")
        assert text == "あるときは全単射"

    def test_kanji_one_is_folded_into_long_vowel_mark(self) -> None:
        """yomitoku が取り違えやすい「一」と「ー」は同じ文字として扱う。"""
        text, _ = build_comparable("一般とコンピュータ")
        assert text == "ー般とコンピユータ"

    def test_small_kana_is_folded_into_normal_kana(self) -> None:
        """yomitoku が取り違えやすい小書きの仮名（「ょ」と「よ」など）は同じ文字として扱う。"""
        text, _ = build_comparable("しょうがっこう")
        assert text == "しようがつこう"

    def test_variant_folding_can_be_disabled(self) -> None:
        text, _ = build_comparable("一般しょう", fold_variants=False)
        assert text == "一般しょう"


class TestExtractGeminiProse:
    """extract_gemini_prose: 数式の位置を印で残した Gemini の地の文を取り出す。"""

    def test_math_is_replaced_by_marker_but_text_in_math_is_kept(self) -> None:
        prose = extract_gemini_prose("式 $x = 1$ と $a \\text{を引くと} b$ です。")
        assert prose == f"式{MATH_MARKER}と{MATH_MARKER}を引くと です。".replace(" ", "")

    def test_code_fence_is_kept_as_text(self) -> None:
        assert extract_gemini_prose("```\n筆算のあいう\n```\n") == "筆算のあいう"


class TestParseAnalysis:
    """parse_analysis: analysis.json の形式を検証する（Fail-Fast）。"""

    def test_missing_paragraphs_raises(self) -> None:
        with pytest.raises(ValueError, match=r"analysis\.json"):
            parse_analysis({"figures": []}, source="page_0001/analysis.json")


class TestComparePage:
    """compare_page: 1 ページ分の食い違いを誤読候補にする。"""

    def test_identical_text_has_no_candidate(self) -> None:
        analysis = _analysis([("有理数を小数で表すと、有限小数になる。", (100, 300, 900, 340))])
        result = compare_page(1, "有理数を小数で表すと，有限小数になる。", analysis)
        assert result.candidates == ()

    def test_replaced_character_is_reported_with_context_and_bbox(self) -> None:
        """Gemini の「服部」と yomitoku の「隈部」の食い違いを、文脈と位置つきで返す。"""
        analysis = _analysis(
            [("©2018 隈部正博", (100, 1272, 260, 1296))],
            words=[("©2018 隈部正博", (102, 1272, 257, 1293))],
        )
        result = compare_page(3, "©2018　服部正博\n", analysis)
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.page == 3
        assert c.kind == "replace"
        assert c.gemini_text == "服"
        assert c.yomitoku_text == "隈"
        assert c.gemini_before.endswith("2018")
        assert c.gemini_after.startswith("部正博")
        assert c.bbox == (102, 1272, 257, 1293)
        assert c.bbox_source == "word"

    def test_gemini_extra_character_is_reported(self) -> None:
        """Gemini だけにある「制」（「無制限」と「無限」）は gemini_only として返す。"""
        analysis = _analysis([("数の並びが無限に繰り返される", (100, 300, 900, 340))])
        result = compare_page(60, "数の並びが無制限に繰り返される", analysis)
        assert [(c.kind, c.gemini_text, c.yomitoku_text) for c in result.candidates] == [
            ("gemini_only", "制", "")
        ]

    def test_yomitoku_extra_character_is_reported(self) -> None:
        analysis = _analysis([("数の並びが無制限に繰り返される", (100, 300, 900, 340))])
        result = compare_page(60, "数の並びが無限に繰り返される", analysis)
        assert [(c.kind, c.gemini_text, c.yomitoku_text) for c in result.candidates] == [
            ("yomitoku_only", "", "制")
        ]
        # Gemini 側は挿入位置を指す（前後の文脈で場所が分かる）
        assert result.candidates[0].gemini_before.endswith("無")
        assert result.candidates[0].gemini_after.startswith("限")

    def test_math_in_gemini_is_ignored(self) -> None:
        """Gemini の数式は比較対象から外れるため、yomitoku が数式を崩して読んでも候補にならない。"""
        analysis = _analysis([("このとき 3/a とおく", (100, 300, 900, 340))])
        result = compare_page(70, "このとき $\\sqrt[4]{a}$ とおく", analysis)
        assert result.candidates == ()

    def test_dash_variants_are_not_candidates(self) -> None:
        """「一」と「ー」の揺れや、数式の分数線の「一」は誤読候補にしない。"""
        analysis = _analysis([("分数一を足すと一般にコンピュータ", (100, 300, 900, 340))])
        result = compare_page(5, "分数を足すと一般にコンピュータ", analysis)
        assert result.candidates == ()

    def test_dash_ignoring_can_be_disabled(self) -> None:
        analysis = _analysis([("分数ーを足す", (100, 300, 900, 340))])
        options = DiffOptions(normalize_variants=False)
        result = compare_page(5, "分数を足す", analysis, options)
        assert len(result.candidates) == 1

    def test_lone_dash_read_in_place_of_a_character_is_ignored(self) -> None:
        """数式のマイナスや分数線の「ー」が日本語の 1 文字の位置に読まれても候補にしない。"""
        analysis = _analysis([("その値は常ーに正である", (100, 300, 900, 340))])
        result = compare_page(151, "その値は常に正である", analysis)
        assert result.candidates == ()

    def test_small_kana_difference_is_not_a_candidate(self) -> None:
        analysis = _analysis([("実感しょう。", (100, 300, 900, 340))])
        result = compare_page(77, "実感しよう。", analysis)
        assert result.candidates == ()

    def test_replace_with_long_text_on_either_side_is_treated_as_misalignment(self) -> None:
        """どちらかが 5 文字を超える置き換えは、読む順序の違いによる対応の取り違えとして候補にしない。"""
        analysis = _analysis(
            [("最初の部分についての説明あいうえおか最後の部分についての説明", (100, 300, 900, 340))]
        )
        result = compare_page(
            1, "最初の部分についての説明さしすせそた最後の部分についての説明", analysis
        )
        assert result.candidates == ()
        assert result.stats.ops_misaligned == 1

    def test_paragraph_inside_figure_is_excluded(self) -> None:
        analysis = _analysis(
            [("図中のラベルあいう", (300, 400, 500, 440))],
            figures=[(250, 350, 800, 600)],
        )
        result = compare_page(61, "本文だけの別の文章です。", analysis)
        assert result.candidates == ()
        assert result.stats.paragraphs_in_figure == 1

    def test_figure_exclusion_can_be_disabled(self) -> None:
        analysis = _analysis(
            [("図中のラベルあいう", (300, 400, 500, 440))],
            figures=[(250, 350, 800, 600)],
        )
        options = DiffOptions(exclude_figures=False, report_unmatched=True)
        result = compare_page(61, "本文だけの別の文章です。", analysis, options)
        assert result.candidates != ()

    def test_running_header_is_excluded(self) -> None:
        """ページ上部の帯にある段落（柱）は比較しない。"""
        analysis = _analysis([("第四章実数の柱", (880, 60, 1000, 95))])
        result = compare_page(70, "本文だけの別の文章です。", analysis)
        assert result.candidates == ()
        assert result.stats.paragraphs_in_header == 1

    def test_short_run_is_ignored(self) -> None:
        """2 文字以下の日本語は偶然の一致・不一致が多いため比較しない。"""
        analysis = _analysis([("あい", (100, 300, 900, 340))])
        result = compare_page(1, "うえ", analysis)
        assert result.candidates == ()
        assert result.stats.runs_short == 1

    def test_run_without_counterpart_is_reported_as_unmatched(self) -> None:
        """Gemini に対応する文章が無い連続は unmatched_run にする。"""
        analysis = _analysis([("全く別の文章が書かれている", (100, 300, 900, 340))])
        result = compare_page(
            1, "かきくけこさしすせそ", analysis, DiffOptions(report_unmatched=True)
        )
        assert [c.kind for c in result.candidates] == ["unmatched_run"]
        assert result.candidates[0].yomitoku_text == "全く別の文章が書かれている"

    def test_paragraph_bbox_is_used_when_no_single_word_contains_the_text(self) -> None:
        analysis = _analysis([("©2018 隈部正博", (100, 1272, 260, 1296))])
        result = compare_page(3, "©2018 服部正博", analysis)
        assert result.candidates[0].bbox == (100, 1272, 260, 1296)
        assert result.candidates[0].bbox_source == "paragraph"

    def test_candidate_to_dict_has_llm_input_fields(self) -> None:
        """後段の LLM 判定への入力形式（ページ・Gemini 側・yomitoku 側）を JSON 化できる。"""
        analysis = _analysis([("©2018 隈部正博", (100, 1272, 260, 1296))])
        data = compare_page(3, "©2018 服部正博", analysis).candidates[0].to_dict()
        assert data["page"] == 3
        assert data["gemini"]["text"] == "服"
        assert data["yomitoku"]["text"] == "隈"
        assert data["yomitoku"]["bbox"] == [100, 1272, 260, 1296]
        assert set(data["gemini"]) == {"text", "context_before", "context_after"}


class TestMathGapNoise:
    """数式の位置（Gemini の区切り）に yomitoku が仮名や記号を読んだ食い違いは候補にしない。"""

    def test_kana_read_in_place_of_math_is_ignored(self) -> None:
        """Gemini では数式の部分を、yomitoku が 1 文字の仮名に読み違えても候補にしない。"""
        analysis = _analysis([("この分数を もで割ればよい", (100, 300, 900, 340))])
        result = compare_page(59, "この分数を $b$ で割ればよい", analysis)
        assert result.candidates == ()

    def test_replaced_character_next_to_math_is_still_reported(self) -> None:
        """数式の隣にあっても、Gemini の日本語を置き換えた食い違いは候補にする。"""
        analysis = _analysis([("2018 隈部正博", (100, 300, 900, 340))])
        result = compare_page(3, "2018 服部正博", analysis)
        assert [(c.gemini_text, c.yomitoku_text) for c in result.candidates] == [("服", "隈")]

    def test_figure_label_merged_into_paragraph_is_ignored(self) -> None:
        """yomitoku が段落に取り込んだ図番号の「図」は候補にしない。"""
        analysis = _analysis([("ここで図4.1の通り", (100, 300, 900, 340))])
        result = compare_page(62, "ここでの通り", analysis)
        assert result.candidates == ()


class TestUnmatchedRuns:
    def test_unmatched_run_is_not_a_candidate_by_default(self) -> None:
        """Gemini に対応箇所が無い連続は、図や記号の誤読が多いため既定では候補にせず件数だけ数える。"""
        analysis = _analysis([("全く別の文章が書かれている", (100, 300, 900, 340))])
        result = compare_page(
            1, "かきくけこさしすせそ", analysis, DiffOptions(report_unmatched=False)
        )
        assert result.candidates == ()
        assert result.stats.runs_unmatched == 1
