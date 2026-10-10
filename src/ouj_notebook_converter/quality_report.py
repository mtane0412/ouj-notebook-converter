"""仕様: 変換結果の品質レポート（現在は KaTeX で描画できない数式の一覧）を作成・出力する。

変換後の全ページの数式を KaTeX で検査し、描画できない数式を「ページ / 数式 / エラー内容」で
一覧化する。出力ディレクトリに次の 2 ファイルを書き出す。
  - quality_report.json: 機械可読（集計とエラー一覧）
  - quality_report.md  : 人が読む形式（ページ順の表）

注意事項:
  - 検査には evaluation.katex.check_katex（Node.js + KaTeX）を再利用する。
    Node.js や katex が無い場合は検査を省略せず KatexCheckError を送出する（Fail-Fast）
  - 数式の抽出は評価と同じ split_markdown を使うため、コードフェンス内の数式は対象外
  - ページ番号は PDF の 1 始まり、数式番号はページ内の出現順（1 始まり）
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ouj_notebook_converter.evaluation.katex import check_katex
from ouj_notebook_converter.evaluation.markdown_parts import Formula, split_markdown
from ouj_notebook_converter.pipeline.types import PageMarkdown

QUALITY_REPORT_JSON = "quality_report.json"
QUALITY_REPORT_MARKDOWN = "quality_report.md"

CheckFn = Callable[[Sequence[Formula]], list[str | None]]


@dataclass(frozen=True)
class KatexError:
    """KaTeX で描画できなかった 1 つの数式。

    Attributes:
        page: PDF のページ番号（1 始まり）。
        formula_index: ページ内での数式の出現順（1 始まり）。
        display: ディスプレイ数式なら True。
        tex: 区切り記号を除いた数式本体。
        message: KaTeX のエラーメッセージ。
    """

    page: int
    formula_index: int
    display: bool
    tex: str
    message: str


@dataclass(frozen=True)
class KatexReport:
    """KaTeX 検査の結果。"""

    total_formulas: int
    errors: tuple[KatexError, ...]


def ensure_katex_available(*, check: CheckFn | None = None) -> None:
    """KaTeX 検査を実行できることを確認する。変換（OCR）を始める前の事前確認に使う。

    Raises:
        KatexCheckError: Node.js や katex が使えない場合。
    """
    (check or check_katex)([Formula(tex="x", display=False)])


def build_katex_report(
    pages: Sequence[PageMarkdown], *, check: CheckFn | None = None
) -> KatexReport:
    """全ページの数式を KaTeX で検査し、描画できない数式を集計する。

    Args:
        pages: 検査するページ（page_index は 0 始まり）。
        check: 数式の検査関数。省略時は evaluation.katex.check_katex（テストで差し替える）。

    Raises:
        KatexCheckError: 検査を実行できなかった場合。
    """
    check_fn = check or check_katex
    total = 0
    errors: list[KatexError] = []
    for page in pages:
        formulas = split_markdown(page.markdown).formulas
        if not formulas:
            continue
        total += len(formulas)
        for index, (formula, message) in enumerate(zip(formulas, check_fn(formulas), strict=True), 1):
            if message is not None:
                errors.append(
                    KatexError(
                        page=page.page_index + 1,
                        formula_index=index,
                        display=formula.display,
                        tex=formula.tex,
                        message=message,
                    )
                )
    return KatexReport(total_formulas=total, errors=tuple(errors))


def create_quality_report(pages: Sequence[PageMarkdown], outdir: Path) -> KatexReport:
    """全ページを検査し、品質レポートを outdir に書き出す（CLI から呼ぶ入口）。"""
    report = build_katex_report(pages)
    write_quality_report(report, outdir)
    return report


def write_quality_report(report: KatexReport, outdir: Path) -> tuple[Path, Path]:
    """品質レポートを outdir に JSON と Markdown で書き出す。

    Returns:
        (JSON のパス, Markdown のパス)。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / QUALITY_REPORT_JSON
    md_path = outdir / QUALITY_REPORT_MARKDOWN
    payload = {
        "katex": {
            "total_formulas": report.total_formulas,
            "error_count": len(report.errors),
            "errors": [
                {
                    "page": e.page,
                    "formula_index": e.formula_index,
                    "display": e.display,
                    "tex": e.tex,
                    "message": e.message,
                }
                for e in report.errors
            ],
        }
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: KatexReport) -> str:
    """人が読む形式のレポートを組み立てる。"""
    lines = [
        "# 変換品質レポート",
        "",
        "## KaTeX で描画できない数式",
        "",
        f"検査した数式: {report.total_formulas} 件 / 描画できない数式: {len(report.errors)} 件",
        "",
    ]
    if not report.errors:
        lines.append("すべて描画できました。")
        return "\n".join(lines) + "\n"
    lines += ["| ページ | 数式番号 | 数式 | エラー内容 |", "| --- | --- | --- | --- |"]
    for e in report.errors:
        lines.append(
            f"| p.{e.page} | {e.formula_index} | {_table_cell(e.tex)} | {_table_cell(e.message)} |"
        )
    return "\n".join(lines) + "\n"


def _table_cell(text: str) -> str:
    """Markdown 表のセルに入れるため、改行を空白にし、区切り文字 | をエスケープしてコードスパンにする。"""
    flat = " ".join(text.split()).replace("|", "\\|").replace("`", "'")
    return f"`{flat}`"
