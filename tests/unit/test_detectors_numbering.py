"""仕様: 式番号・例番号・脚注番号の連番検査（detectors/numbering.py）のテスト。

定義と参照の区別、欠番・重複・順序の逆転・未定義参照の検出、章の追跡、CLI の動作を確かめる。
テストデータは検査の挙動が分かる短い架空の文であり、教材の本文は含まない。
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.detectors.numbering import (
    Occurrence,
    app,
    check_numbering,
    extract_occurrences,
)


def _defs(markdown: str, page: int = 1, chapter: int | None = 1) -> list[tuple[str, int, int]]:
    occurrences, _ = extract_occurrences(page, markdown, chapter)
    return [(o.kind, o.chapter, o.number) for o in occurrences if o.is_definition]


def _refs(markdown: str, page: int = 1, chapter: int | None = 1) -> list[tuple[str, int, int]]:
    occurrences, _ = extract_occurrences(page, markdown, chapter)
    return [(o.kind, o.chapter, o.number) for o in occurrences if not o.is_definition]


class TestEquationNumbers:
    def test_line_end_equation_number_is_definition(self) -> None:
        """行末の式番号は定義になる"""
        assert _defs("$$ a + b = b + a \\qquad (1.2) $$") == [("式", 1, 2)]

    def test_tag_is_definition(self) -> None:
        """tagは定義になる"""
        assert _defs("$$ a + b = b + a \\tag{2.10} $$") == [("式", 2, 10)]

    def test_array_row_end_equation_number_is_definition(self) -> None:
        """配列の行末にある式番号は定義になる"""
        assert _defs("\\text{加法の交換法則} & (1.4) \\\\") == [("式", 1, 4)]

    def test_mid_sentence_equation_number_is_reference(self) -> None:
        """地の文の途中にある式番号は参照になる"""
        markdown = "このとき (2.9) より，$a = 0$ が成り立つ。"
        assert _defs(markdown) == []
        assert _refs(markdown) == [("式", 2, 9)]

    def test_equation_number_followed_by_period_is_reference(self) -> None:
        """文末の句点が続く式番号は参照になる"""
        assert _refs("これは (1.5)。") == [("式", 1, 5)]

    def test_out_of_range_chapter_is_not_equation_number(self) -> None:
        """小数を含む括弧は章番号の範囲外なら式番号とみなさない"""
        assert _refs("点 (0.5) と (99.1) を結ぶ。") == []

    def test_code_block_is_ignored(self) -> None:
        """コードブロック内の式番号は無視する"""
        assert _defs("```\n(1.2)\n```") == []

    def test_standalone_label_in_code_block_is_definition(self) -> None:
        """文字で描いた図のコードブロックに残った図番号は定義になる"""
        assert _defs("```\n  ＜ 3通り\n     図 11.1\n```", chapter=11) == [("図", 11, 1)]

    def test_page_annotated_equation_number_is_reference(self) -> None:
        """ページ番号の添え字が付いた式番号は参照になる"""
        markdown = "(1.17)$_{[\\text{p.17}]}$ と (3.10) $_{\\text{[p.52]}}$ を使う。"
        assert _defs(markdown) == []
        assert _refs(markdown) == [("式", 1, 17), ("式", 3, 10)]


class TestLabels:
    def test_bold_label_is_definition(self) -> None:
        """太字のラベルは定義になる"""
        assert _defs("**例 2.3**\n\n**練習 2.5**\n\n**コメント 3.1 (C)**") == [
            ("例", 2, 3),
            ("練習", 2, 5),
            ("コメント", 3, 1),
        ]

    def test_standalone_figure_label_is_definition(self) -> None:
        """行頭に単独で置かれた図番号は定義になる"""
        assert _defs("図 2.4") == [("図", 2, 4)]

    def test_html_wrapped_figure_label_is_definition(self) -> None:
        """HTMLタグで包まれた図番号は定義になる"""
        assert _defs('<div align="center">図 2.7</div>') == [("図", 2, 7)]

    def test_mention_in_prose_is_reference(self) -> None:
        """本文中の例や図の言及は参照になる"""
        markdown = "図 1.1 より，正の数が大きい。例 9.2 参照。"
        assert _defs(markdown) == []
        assert _refs(markdown) == [("図", 1, 1), ("例", 9, 2)]


class TestFootnotes:
    def test_line_start_footnote_mark_is_definition(self) -> None:
        """行頭の脚注記号は定義になる"""
        markdown = "$*1$ 補足の説明。\n*2 別の補足。\n$*3\\quad$ さらに補足。"
        assert _defs(markdown, chapter=4) == [("脚注", 4, 1), ("脚注", 4, 2), ("脚注", 4, 3)]

    def test_line_start_superscript_footnote_mark_is_definition(self) -> None:
        """行頭の上付き形式の脚注記号は定義になる"""
        markdown = "$^{*1}$ 補足の説明。"
        assert _defs(markdown, chapter=3) == [("脚注", 3, 1)]
        assert _refs(markdown, chapter=3) == []

    def test_line_start_braced_superscript_footnote_mark_is_definition(self) -> None:
        """行頭の空の波括弧付き上付き形式の脚注記号も定義になる"""
        assert _defs("${}^{*13}$ 詳しくは別の節を参照。", chapter=4) == [("脚注", 4, 13)]

    def test_superscript_footnote_mark_is_reference(self) -> None:
        """上付きの脚注記号は参照になる"""
        markdown = (
            "素因数分解と言う$^{*1}$。次に ${}^{*2}$ と <sup>\\*3</sup> と \\text{である}^{*4}。"
        )
        assert _refs(markdown, chapter=1) == [
            ("脚注", 1, 1),
            ("脚注", 1, 2),
            ("脚注", 1, 3),
            ("脚注", 1, 4),
        ]

    def test_footnote_without_chapter_is_chapter_zero(self) -> None:
        """章が分からない脚注は0章として扱う"""
        assert _defs("$*1$ 補足。", chapter=None) == [("脚注", 0, 1)]


class TestChapterTracking:
    def test_chapter_heading_switches_chapter(self) -> None:
        """章見出しで現在の章が切り替わる"""
        markdown = "# 4 実数\n\n本文$^{*1}$。\n\n$*1$ 補足。"
        occurrences, chapter = extract_occurrences(1, markdown, 3)
        assert chapter == 4
        assert {o.chapter for o in occurrences} == {4}

    def test_section_heading_switches_chapter(self) -> None:
        """節見出しでも現在の章が切り替わる"""
        _, chapter = extract_occurrences(1, "## 5.2 方程式", 4)
        assert chapter == 5

    def test_toc_entry_is_not_chapter_heading(self) -> None:
        """目次の項目はページ番号で終わるので章見出しとみなさない"""
        _, chapter = extract_occurrences(1, "# 6 図形の性質　　96", 2)
        assert chapter == 2


def _numbered(chapter: int, numbers: list[int]) -> str:
    return "\n".join(f"$$ x = {n} \\tag{{{chapter}.{n}}} $$" for n in numbers)


class TestCheckNumbering:
    def test_complete_sequence_has_no_findings(self) -> None:
        """連番が揃っていれば指摘は0件"""
        report = check_numbering([(1, _numbered(1, [1, 2, 3]))])
        assert report.findings == []

    def test_missing_number_reports_surrounding_pages(self) -> None:
        """欠番を前後のページ付きで報告する"""
        pages = [(10, _numbered(1, [1, 2])), (11, _numbered(1, [4, 5]))]
        report = check_numbering(pages)
        (finding,) = report.findings
        assert (finding.type, finding.kind, finding.chapter, finding.number) == (
            "missing",
            "式",
            1,
            3,
        )
        assert (finding.after_page, finding.before_page) == (10, 11)

    def test_missing_number_attaches_references(self) -> None:
        """欠番に対する参照があれば参照元を添える"""
        pages = [(10, _numbered(1, [1, 2, 4])), (11, "(1.3) より成り立つ。")]
        (finding,) = check_numbering(pages).findings
        assert finding.type == "missing"
        assert [(r.page, r.line) for r in finding.references] == [(11, 1)]

    def test_duplicate_definition_is_reported(self) -> None:
        """重複した定義を報告する"""
        pages = [(10, _numbered(1, [1, 2])), (12, _numbered(1, [2, 3]))]
        (finding,) = check_numbering(pages).findings
        assert (finding.type, finding.number) == ("duplicate", 2)
        assert [d.page for d in finding.definitions] == [10, 12]

    def test_order_reversal_is_reported(self) -> None:
        """順序の逆転を報告する"""
        report = check_numbering([(10, _numbered(1, [1, 3, 2, 4]))])
        (finding,) = report.findings
        assert (finding.type, finding.number) == ("order", 2)

    def test_undefined_reference_is_reported(self) -> None:
        """定義のない番号への参照を未定義参照として報告する"""
        pages = [(10, _numbered(1, [1, 2])), (11, "(1.9) を使う。")]
        (finding,) = check_numbering(pages).findings
        assert (finding.type, finding.number) == ("undefined_reference", 9)

    def test_cross_chapter_reference_resolves(self) -> None:
        """章をまたぐ参照は参照先の章の定義で解決する"""
        pages = [(10, _numbered(1, [1, 2])), (11, "# 2 式と計算\n\n(1.2) の再掲。")]
        assert check_numbering(pages).findings == []

    def test_chapters_are_checked_independently(self) -> None:
        """章ごとに独立して検査する"""
        pages = [(10, _numbered(1, [1, 2])), (11, _numbered(2, [1, 3]))]
        (finding,) = check_numbering(pages).findings
        assert (finding.chapter, finding.number) == (2, 2)

    def test_footnotes_are_checked_per_chapter_heading(self) -> None:
        """脚注は章の見出しごとに検査する"""
        pages = [
            (10, "# 4 実数\n\n$*1$ 一つ目。\n$*3$ 三つ目。"),
            (11, "# 5 方程式\n\n$*1$ 一つ目。"),
        ]
        (finding,) = check_numbering(pages).findings
        assert (finding.kind, finding.chapter, finding.number) == ("脚注", 4, 2)

    def test_summary_counts_per_kind(self) -> None:
        """集計は種別ごとの定義数と指摘数を返す"""
        report = check_numbering([(10, _numbered(1, [1, 3]))])
        assert report.definition_counts == {"式": 2}
        assert report.finding_counts() == {"missing": 1}


class TestOccurrence:
    def test_occurrence_has_page_and_line(self) -> None:
        """出現箇所はページと行を持つ"""
        occurrences, _ = extract_occurrences(7, "本文\n$$ x \\tag{1.1} $$", 1)
        assert occurrences == [
            Occurrence(kind="式", chapter=1, number=1, page=7, line=2, is_definition=True)
        ]


class TestCli:
    def test_ページキャッシュを検査してJSONを書き出す(self, tmp_path: Path) -> None:
        for page, body in [(1, _numbered(1, [1, 2])), (2, _numbered(1, [4]))]:
            page_dir = tmp_path / f"page_{page:04d}"
            page_dir.mkdir()
            (page_dir / "raw.md").write_text(body, encoding="utf-8")
        out = tmp_path / "result.json"

        result = CliRunner().invoke(app, ["--pred", str(tmp_path), "--json", str(out)])

        assert result.exit_code == 0, result.output
        data = json.loads(out.read_text(encoding="utf-8"))
        assert [(f["type"], f["number"]) for f in data["findings"]] == [("missing", 3)]
        assert data["definition_counts"] == {"式": 3}

    def test_存在しないディレクトリはエラー終了する(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["--pred", str(tmp_path / "none")])
        assert result.exit_code == 1
