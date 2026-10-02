"""仕様: 評価スクリプトの CLI。評価セットと評価対象を比較し、指標を表で表示する。

使い方:
  uv run python -m ouj_notebook_converter.evaluation \\
      --truth <評価セットのディレクトリ> --pred <出力 or キャッシュのディレクトリ> [--json 結果.json]

ounc 本体は入力 PDF を位置引数に取る単一コマンドのため、サブコマンドにせず独立した CLI とする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.evaluation.katex import KatexCheckError, check_katex
from ouj_notebook_converter.evaluation.metrics import PageScores
from ouj_notebook_converter.evaluation.report import (
    DatasetReport,
    PageReport,
    SummaryReport,
    evaluate_dataset,
)

app = typer.Typer(add_completion=False)


@app.command()
def evaluate(
    truth: Annotated[
        Path, typer.Option("--truth", help="評価セット（manifest.json と正解 page_NNNN.md）")
    ],
    pred: Annotated[
        Path, typer.Option("--pred", help="評価対象（--no-combine 出力 または ページキャッシュ）")
    ],
    json_path: Annotated[
        Path | None, typer.Option("--json", help="評価結果を書き出す JSON ファイル")
    ] = None,
) -> None:
    """評価セットの正解と評価対象の出力を比較し、CER・数式/見出しの一致率・KaTeX エラー件数を表示する。"""
    try:
        report = evaluate_dataset(truth, pred, katex_checker=check_katex)
    except (FileNotFoundError, ValueError, KatexCheckError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    console = Console()
    console.print(_build_table(report))
    if json_path is not None:
        json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"評価結果を書き出しました: {json_path}")


def _build_table(report: DatasetReport) -> Table:
    """ページ別・層別・全体の指標を 1 つの表にまとめる。"""
    table = Table(title="評価結果（CER は低いほど良い、F1 は高いほど良い）")
    for column in ("対象", "層", "CER", "数式F1", "数式CER", "見出しF1", "KaTeXエラー"):
        table.add_column(column)
    for row in report.pages:
        table.add_row(f"p.{row.page}", row.category, *_score_cells(row))
    table.add_section()
    for category, summary in report.by_category.items():
        table.add_row(f"{summary.page_count} ページ", category, *_score_cells(summary))
    table.add_section()
    table.add_row(f"{report.total.page_count} ページ", "全体", *_score_cells(report.total))
    return table


def _score_cells(item: PageReport | SummaryReport) -> list[str]:
    scores: PageScores = item.scores
    katex_errors = (
        len(item.katex_errors) if isinstance(item, PageReport) else item.katex_error_count
    )
    return [
        f"{scores.cer.cer:.2%}",
        f"{scores.math.f1:.3f} ({scores.math.matched}/{scores.math.truth_count})",
        f"{scores.math_cer.cer:.2%}",
        f"{scores.headings.f1:.3f} ({scores.headings.matched}/{scores.headings.truth_count})",
        str(katex_errors),
    ]
