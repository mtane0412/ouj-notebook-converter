"""仕様: 等式の検算による誤読候補検出（issue #19）の CLI。

使い方:
  uv run --extra verify python -m ouj_notebook_converter.detectors \\
      --pred <出力 or キャッシュのディレクトリ> [--truth <評価セットのディレクトリ>] [--json 結果.json]

- 評価対象のディレクトリにある全ページの数式を検算し、検算結果の内訳と誤読候補の一覧を表示する
- --truth を指定すると、評価セットのページについて誤読候補を正解と照らし合わせ、適合率・再現率を表示する
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.detectors.equation_check import (
    LinkStatus,
    MarkdownCheck,
    check_markdown,
)
from ouj_notebook_converter.detectors.scoring import DetectionScore, score_page, sum_scores
from ouj_notebook_converter.evaluation.dataset import (
    list_prediction_pages,
    load_manifest,
    read_prediction,
    read_truth,
)

app = typer.Typer(add_completion=False)


@app.command()
def equation_check(
    pred: Annotated[
        Path, typer.Option("--pred", help="評価対象（--no-combine 出力 または ページキャッシュ）")
    ],
    truth: Annotated[
        Path | None,
        typer.Option("--truth", help="評価セット（指定すると適合率・再現率を計算する）"),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="検出結果を書き出す JSON ファイル")
    ] = None,
) -> None:
    """数式の等式・不等式を検算し、成り立たない式を誤読候補として表示する。"""
    try:
        # 検算と適合率の計算で同じ Markdown を使う（読み直すと数式の番号がずれうるため）
        predictions = {page: read_prediction(pred, page) for page in list_prediction_pages(pred)}
        checks = {page: check_markdown(markdown) for page, markdown in predictions.items()}
        result = _summarize(checks)
        if truth is not None:
            result["scores"] = _score(truth, pred, predictions, checks)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    console = Console()
    console.print(_status_table(result))
    console.print(_violation_table(result["violations"]))
    if "scores" in result:
        console.print(_score_table(result["scores"]))
    if json_path is not None:
        try:
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(code=1) from e
        console.print(f"検出結果を書き出しました: {json_path}")


def _summarize(checks: dict[int, MarkdownCheck]) -> dict[str, Any]:
    """全ページの検算結果を、件数の内訳と誤読候補の一覧にまとめる。"""
    counts = Counter(r.status for check in checks.values() for r in check.links)
    violations = [
        {
            "page": page,
            "formula": r.formula,
            "lhs": r.link.lhs,
            "relation": r.link.relation,
            "rhs": r.link.rhs,
            "following_text": r.link.following_text,
        }
        for page, check in checks.items()
        for r in check.links
        if r.status == LinkStatus.VIOLATED
    ]
    return {
        "page_count": len(checks),
        "formula_count": sum(c.formula_count for c in checks.values()),
        "skipped_formula_count": sum(c.skipped_formula_count for c in checks.values()),
        "status_counts": {status.value: counts[status] for status in LinkStatus},
        "violations": violations,
    }


def _score(
    truth: Path, pred: Path, predictions: dict[int, str], checks: dict[int, MarkdownCheck]
) -> dict[str, Any]:
    """評価セットのページについて、誤読候補を正解と照らし合わせる。

    predictions は check_markdown に渡したものと同じ Markdown（formula_index の対応を保つため）。
    """
    pages: list[dict[str, Any]] = []
    scores: list[DetectionScore] = []
    for entry in load_manifest(truth):
        if entry.page not in checks:
            raise FileNotFoundError(
                f"評価セットのページ {entry.page} が評価対象にありません: {pred}"
            )
        flagged = {
            r.formula_index for r in checks[entry.page].links if r.status == LinkStatus.VIOLATED
        }
        score = score_page(read_truth(truth, entry.page), predictions[entry.page], flagged)
        scores.append(score)
        pages.append({"page": entry.page, "category": entry.category, **score.to_dict()})
    return {"pages": pages, "total": sum_scores(scores).to_dict()}


def _status_table(result: dict[str, Any]) -> Table:
    table = Table(
        title=f"検算結果の内訳（{result['page_count']} ページ・数式 {result['formula_count']} 件）"
    )
    table.add_column("区分")
    table.add_column("件数", justify="right")
    for status, count in result["status_counts"].items():
        table.add_row(f"リンク: {status}", str(count))
    table.add_row("数式: 対象外の環境（array・cases など）", str(result["skipped_formula_count"]))
    return table


def _violation_table(violations: list[dict[str, Any]]) -> Table:
    table = Table(title=f"誤読候補（{len(violations)} 件）")
    for column in ("ページ", "左辺", "関係", "右辺", "数式"):
        table.add_column(column)
    for v in violations:
        table.add_row(f"p.{v['page']}", v["lhs"], v["relation"], v["rhs"], v["formula"])
    return table


def _score_table(scores: dict[str, Any]) -> Table:
    table = Table(title="評価セットでの適合率・再現率（数式単位）")
    for column in ("対象", "誤読候補", "うち誤読", "誤読", "適合率", "再現率"):
        table.add_column(column)
    rows = [(f"p.{p['page']}", p) for p in scores["pages"] if p["flagged"] or p["misread"]]
    for label, s in [*rows, ("全体", scores["total"])]:
        table.add_row(
            label,
            str(s["flagged"]),
            str(s["true_positive"]),
            str(s["misread"]),
            f"{s['precision']:.3f}",
            f"{s['recall']:.3f}",
        )
    return table
