"""仕様: quality_report モジュール（変換結果の KaTeX 描画可否レポート）のユニットテスト。

KaTeX の検査（check_katex）はフェイクに差し替え、ページ・数式番号の対応づけ、
JSON / Markdown 形式の出力、Node.js が無いときの Fail-Fast を検証する。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from ouj_notebook_converter.evaluation.katex import KatexCheckError
from ouj_notebook_converter.evaluation.markdown_parts import Formula
from ouj_notebook_converter.pipeline.types import PageMarkdown
from ouj_notebook_converter.quality_report import (
    QUALITY_REPORT_JSON,
    QUALITY_REPORT_MARKDOWN,
    build_katex_report,
    ensure_katex_available,
    write_quality_report,
)

CheckFn = Callable[[Sequence[Formula]], list[str | None]]


def _fake_check(errors: dict[str, str]) -> CheckFn:
    """tex が errors のキーに一致する数式だけをエラーにするフェイク検査関数を返す。"""

    def _check(formulas: Sequence[Formula]) -> list[str | None]:
        return [errors.get(f.tex) for f in formulas]

    return _check


def _page(index0: int, markdown: str) -> PageMarkdown:
    return PageMarkdown(page_index=index0, markdown=markdown)


class TestBuildKatexReport:
    """build_katex_report の集計。"""

    def test_描画できない数式をページと数式番号つきで列挙する(self) -> None:
        pages = [
            _page(0, "正常な式 $x+1$ です。"),
            _page(14, "$$a = b$$\n\n壊れた式 $\\cline{1-2}$ です。"),
        ]
        report = build_katex_report(
            pages, check=_fake_check({"\\cline{1-2}": "Undefined control sequence: \\cline"})
        )

        assert report.total_formulas == 3
        assert len(report.errors) == 1
        error = report.errors[0]
        assert error.page == 15  # 0-origin の page_index を 1-origin に直す
        assert error.formula_index == 2  # ページ内で 2 番目の数式
        assert error.tex == "\\cline{1-2}"
        assert error.display is False
        assert error.message == "Undefined control sequence: \\cline"

    def test_数式が無ければ検査を呼ばない(self) -> None:
        def _never(formulas: Sequence[Formula]) -> list[str | None]:
            raise AssertionError("数式が無いので検査は呼ばれないはず")

        report = build_katex_report([_page(0, "数式のないページです。")], check=_never)

        assert report.total_formulas == 0
        assert report.errors == ()

    def test_コードフェンス内の数式は検査しない(self) -> None:
        report = build_katex_report(
            [_page(0, "```\n$\\cline{1-2}$\n```\n")],
            check=_fake_check({"\\cline{1-2}": "エラー"}),
        )

        assert report.total_formulas == 0
        assert report.errors == ()


class TestWriteQualityReport:
    """write_quality_report の出力。"""

    def test_JSONと人が読めるMarkdownを出力する(self, tmp_path: Path) -> None:
        report = build_katex_report(
            [_page(33, "筆算 $$\\multicolumn{2}{c}{x}$$")],
            check=_fake_check({"\\multicolumn{2}{c}{x}": "Undefined control sequence"}),
        )

        write_quality_report(report, tmp_path)

        data = json.loads((tmp_path / QUALITY_REPORT_JSON).read_text(encoding="utf-8"))
        assert data["katex"]["total_formulas"] == 1
        assert data["katex"]["error_count"] == 1
        assert data["katex"]["errors"][0] == {
            "page": 34,
            "formula_index": 1,
            "display": True,
            "tex": "\\multicolumn{2}{c}{x}",
            "message": "Undefined control sequence",
        }
        text = (tmp_path / QUALITY_REPORT_MARKDOWN).read_text(encoding="utf-8")
        assert "p.34" in text
        assert "Undefined control sequence" in text

    def test_エラーが無くてもレポートを出力する(self, tmp_path: Path) -> None:
        report = build_katex_report([_page(0, "$x$")], check=_fake_check({}))

        write_quality_report(report, tmp_path)

        data = json.loads((tmp_path / QUALITY_REPORT_JSON).read_text(encoding="utf-8"))
        assert data["katex"]["error_count"] == 0
        text = (tmp_path / QUALITY_REPORT_MARKDOWN).read_text(encoding="utf-8")
        assert "すべて描画できました" in text

    def test_表のセル区切りと衝突しないようエスケープする(self, tmp_path: Path) -> None:
        report = build_katex_report([_page(0, "$a|b$")], check=_fake_check({"a|b": "エラー"}))

        write_quality_report(report, tmp_path)

        text = (tmp_path / QUALITY_REPORT_MARKDOWN).read_text(encoding="utf-8")
        assert "a\\|b" in text


class TestEnsureKatexAvailable:
    """変換開始前の事前確認。"""

    def test_検査できれば何も起きない(self) -> None:
        ensure_katex_available(check=lambda formulas: [None for _ in formulas])

    def test_Nodeが無ければKatexCheckErrorを送出する(self) -> None:
        def _broken(formulas: Sequence[Formula]) -> list[str | None]:
            raise KatexCheckError("Node.js（node コマンド）が見つかりません")

        with pytest.raises(KatexCheckError, match=r"Node\.js"):
            ensure_katex_available(check=_broken)
