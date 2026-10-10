"""仕様: judge.candidates のユニットテスト。

各検出器の誤読候補を共通形式（JudgeCandidate）に正規化し、
OCR Markdown（raw.md）上の該当行を抜粋として取り出すことをテストする。
"""

import json
from pathlib import Path

import pytest

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import Candidate as DiffCandidate
from ouj_notebook_converter.detectors.cross_ocr_diff.detect import parse_analysis
from ouj_notebook_converter.judge.candidates import (
    _locate_diff_line,
    collect_candidates,
    cross_ocr_candidates,
    equation_candidates,
    find_excerpt,
    rare_word_candidates,
    read_raw_pages,
)
from ouj_notebook_converter.judge.models import JudgeCandidate, load_candidates, save_candidates

_BROKEN_EQUATION_PAGE = (
    "例 4.3 を考える。\n\n$$(\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4$$\n\nしたがって成り立つ。\n"
)


def test_find_excerpt_は該当行と前後1行を返す() -> None:
    raw = "一行目\n二行目\n三行目に無限がある\n四行目\n五行目"
    excerpt = find_excerpt(raw, "無限")
    assert excerpt is not None
    assert excerpt.text == "二行目\n三行目に無限がある\n四行目"
    assert excerpt.line_index == 2


def test_find_excerpt_は複数行にまたがる数式の全行を含める() -> None:
    raw = "前の文\n$$\na = b\n\\\\\nc = d\n$$\n後の文"
    excerpt = find_excerpt(raw, "a = b\n\\\\\nc = d")
    assert excerpt is not None
    assert "c = d" in excerpt.text
    assert excerpt.line_index == 2


def test_find_excerpt_は見つからなければNoneを返す() -> None:
    assert find_excerpt("本文", "存在しない文字列") is None


def test_equation_candidates_は成り立たない等式を候補にする() -> None:
    candidates = equation_candidates({70: _BROKEN_EQUATION_PAGE})
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "equation_check"
    assert candidate.page == 70
    assert "(\\sqrt[3]{a})^4" in candidate.text
    assert "$$(\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4$$" in candidate.excerpt
    assert candidate.excerpt_scope == "lines"


def test_rare_word_candidates_は低頻度語を候補にする() -> None:
    pages = {i: "著者は隈部正博です。\n" * 2 for i in range(1, 4)}
    pages[9] = "©2018　服部正博\n"
    candidates = rare_word_candidates(pages, skip_index=True)
    assert [c.text for c in candidates] == ["服部正博"]
    assert candidates[0].page == 9
    assert "隈部正博" in candidates[0].hint


def test_cross_ocr_candidates_は食い違いと_yomitoku_の位置を持つ(tmp_path: Path) -> None:
    pages = {3: "著者は服部正博です。\n"}
    analysis_dir = tmp_path / "page_0003"
    analysis_dir.mkdir()
    paragraph = {"box": [100, 300, 900, 340], "contents": "著者は隈部正博です。", "role": None}
    (analysis_dir / "analysis.json").write_text(
        json.dumps(
            {"paragraphs": [paragraph], "figures": [], "tables": [], "words": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    candidates = cross_ocr_candidates(pages, tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "cross_ocr_diff"
    assert candidate.kind == "replace"
    assert candidate.text == "服"
    assert candidate.yomitoku_text == "隈"
    assert candidate.bbox == (100, 300, 900, 340)
    assert candidate.location == "yomitoku_paragraph"
    assert "著者は服部正博です。" in candidate.excerpt


def test_collect_candidates_はIDを振り_JSONに往復できる(tmp_path: Path) -> None:
    candidates = collect_candidates(
        {70: _BROKEN_EQUATION_PAGE}, yomitoku_dir=None, sources={"equation_check"}
    )
    assert [c.id for c in candidates] == ["p0070-equation_check-1"]
    path = tmp_path / "candidates.json"
    save_candidates(path, candidates)
    assert load_candidates(path) == candidates


def test_load_candidates_は_format_version_が違えばエラーにする(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"format_version": 99, "candidates": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_candidates(path)


def test_cross_ocr_を要求して_yomitoku_が無ければエラーにする() -> None:
    with pytest.raises(ValueError, match="yomitoku"):
        collect_candidates({1: "本文"}, yomitoku_dir=None, sources={"cross_ocr_diff"})


def test_read_raw_pages_は後処理せずraw_mdを読む(tmp_path: Path) -> None:
    (tmp_path / "page_0001").mkdir()
    (tmp_path / "page_0001" / "raw.md").write_text("# 見出し\n", encoding="utf-8")
    assert read_raw_pages(tmp_path) == {1: "# 見出し\n"}


def test_JudgeCandidate_は辞書に往復できる() -> None:
    candidate = JudgeCandidate(
        id="p0001-rare_word-1",
        page=1,
        source="rare_word",
        kind="rare_word",
        text="服部",
        hint="ヒント",
        excerpt="抜粋",
        excerpt_scope="lines",
        line_index=0,
    )
    assert JudgeCandidate.from_dict(candidate.to_dict()) == candidate


def test_cross_ocr_candidates_は数式を含む行も抜粋として見つける(tmp_path: Path) -> None:
    raw = "前置きの行です。\n\n著者は服部正博です。$x$ と考える。\n"
    analysis_dir = tmp_path / "page_0001"
    analysis_dir.mkdir()
    paragraph = {
        "box": [100, 300, 900, 340],
        "contents": "著者は隈部正博です。x と考える。",
        "role": None,
    }
    (analysis_dir / "analysis.json").write_text(
        json.dumps(
            {"paragraphs": [paragraph], "figures": [], "tables": [], "words": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    candidates = cross_ocr_candidates({1: raw}, tmp_path)
    assert candidates, "食い違いが検出されるはず"
    # 文脈に「〔数式〕」を含んでも、数式を含む行が見つかる（ページ全体へのフォールバックにならない）
    assert any("〔数式〕" in c.context_after for c in candidates)
    assert all(c.excerpt_scope == "lines" for c in candidates)


def test_cross_ocr_の数式ラベルを含む文脈でも行を見つける() -> None:
    """検出器の文脈は数式を「〔数式〕」と書く。行の地の文にも同じ置き換えをして照合する。"""
    raw = "前置き。\n\nしかし，集合 $A$ と集合 $B$ は，どちらも $a$ と $b$ という 2 つの要素からなる。\n"
    diff = DiffCandidate(
        page=1,
        kind="gemini_only",
        gemini_text="〔数式〕と〔数式〕",
        gemini_before="し,集合〔数式〕と集合〔数式〕は,どちらも",
        gemini_after="という2つの要素からなる集合で",
        gemini_span=(1, 2),
        yomitoku_text="",
        yomitoku_before="",
        yomitoku_after="",
        bbox=(0, 0, 1, 1),
        bbox_source="paragraph",
    )
    excerpt = _locate_diff_line(raw, diff, None)
    assert excerpt is not None
    assert excerpt.line_index == 2


def test_cross_ocr_の文脈が複数行に一致するときは_bbox_で行を決める() -> None:
    """似た行が複数あるとき、先頭の行を選ぶと別の行を LLM に見せてしまう。bbox の位置で見分ける。"""
    raw = (
        "つまり，左辺の積の対数が，右辺では各数の対数の和に変わることになる。\n\n"
        "つまり，左辺の商の対数が，右辺では各数の対数の差に変わることになる。\n"
    )
    analysis = parse_analysis(
        {
            "paragraphs": [
                {
                    "box": [100, 100, 900, 160],
                    "contents": "つまり，左辺の積の対数が，右辺では各数の対数の和に変わることになる。",
                    "role": None,
                },
                {
                    "box": [100, 400, 900, 460],
                    "contents": "つまり，左辺の商の対数が，右辺では各数の対数の差に変わることになる。",
                    "role": None,
                },
            ],
            "figures": [],
            "tables": [],
            "words": [],
        },
        source="テスト",
    )
    diff = DiffCandidate(
        page=1,
        kind="replace",
        gemini_text="つまり",
        gemini_before="",
        gemini_after="",
        gemini_span=(0, 3),
        yomitoku_text="つま",
        yomitoku_before="",
        yomitoku_after="",
        bbox=(100, 410, 300, 450),
        bbox_source="word",
    )
    excerpt = _locate_diff_line(raw, diff, analysis)
    assert excerpt is not None
    assert excerpt.line_index == 2


def test_cross_ocr_の文脈が複数行に一致し_bbox_でも決まらなければ_None() -> None:
    raw = "つまり，左辺の積の対数が。\n\nつまり，左辺の商の対数が。\n"
    diff = DiffCandidate(
        page=1,
        kind="replace",
        gemini_text="つまり",
        gemini_before="",
        gemini_after="",
        gemini_span=(0, 3),
        yomitoku_text="つま",
        yomitoku_before="",
        yomitoku_after="",
        bbox=(0, 0, 1, 1),
        bbox_source="word",
    )
    assert _locate_diff_line(raw, diff, None) is None
