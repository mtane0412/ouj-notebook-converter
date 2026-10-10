"""仕様: Gemini と yomitoku の食い違いから地の文の誤読候補を抽出する検出器（issue #23）の CLI。

使い方:
  uv run python -m ouj_notebook_converter.detectors.cross_ocr_diff \\
      --gemini <Gemini の出力 or キャッシュ> --yomitoku <yomitoku のキャッシュ> \\
      [--truth <評価セット>] [--json 結果.json]

- 両方のディレクトリの全ページを比較し、誤読候補の件数・内訳・一覧を表示する
- --truth を指定すると、評価セットのページについて適合率・再現率を表示する
- --json は後段の LLM 判定（issue #24）への入力にもなる（形式は README を参照）
- ノイズ除去は個別に無効化できる（除去の効果を測るため）
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    DEFAULT_MIN_RUN_LENGTH,
    DiffOptions,
    PageResult,
    PageStats,
    compare_pages,
)
from ouj_notebook_converter.detectors.cross_ocr_diff.scoring import (
    CrossOcrScore,
    find_gemini_misreads,
    score_page,
    sum_scores,
)
from ouj_notebook_converter.evaluation.dataset import load_manifest, read_prediction, read_truth

# JSON の形式のバージョン（後段の LLM 判定が形式の変更を検知するため）
FORMAT_VERSION = 1

app = typer.Typer(add_completion=False)


@app.command()
def cross_ocr_diff(
    gemini: Annotated[
        Path, typer.Option("--gemini", help="Gemini の出力（--no-combine）またはページキャッシュ")
    ],
    yomitoku: Annotated[
        Path, typer.Option("--yomitoku", help="yomitoku のページキャッシュ（analysis.json を含む）")
    ],
    truth: Annotated[
        Path | None,
        typer.Option("--truth", help="評価セット（指定すると適合率・再現率を計算する）"),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="検出結果を書き出す JSON ファイル")
    ] = None,
    exclude_figures: Annotated[
        bool, typer.Option("--exclude-figures/--no-exclude-figures", help="図の領域の段落を除く")
    ] = True,
    exclude_header: Annotated[
        bool,
        typer.Option("--exclude-header/--no-exclude-header", help="ページ上部の帯（柱）を除く"),
    ] = True,
    normalize_variants: Annotated[
        bool,
        typer.Option(
            "--normalize-variants/--no-normalize-variants",
            help="「一」と「ー」、小書きの仮名などの文字種の揺れを除く",
        ),
    ] = True,
    min_run_length: Annotated[
        int, typer.Option("--min-run-length", min=1, help="比較する日本語の連続の最小文字数")
    ] = DEFAULT_MIN_RUN_LENGTH,
    report_unmatched: Annotated[
        bool,
        typer.Option(
            "--report-unmatched/--no-report-unmatched",
            help="Gemini に対応箇所が無い連続も候補にする",
        ),
    ] = False,
) -> None:
    """Gemini と yomitoku の日本語部分を比較し、食い違いを誤読候補として表示する。"""
    options = DiffOptions(
        min_run_length=min_run_length,
        exclude_figures=exclude_figures,
        exclude_header=exclude_header,
        normalize_variants=normalize_variants,
        report_unmatched=report_unmatched,
    )
    try:
        results = compare_pages(gemini, yomitoku, options)
        output = _build_output(gemini, yomitoku, options, results)
        if truth is not None:
            output["scores"] = _score(truth, gemini, results)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    console = Console()
    console.print(_stats_table(output))
    console.print(_candidate_table(results))
    if "scores" in output:
        console.print(_score_table(output["scores"]))
    if json_path is not None:
        try:
            json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(code=1) from e
        console.print(f"検出結果を書き出しました: {json_path}")


def _build_output(
    gemini: Path, yomitoku: Path, options: DiffOptions, results: list[PageResult]
) -> dict[str, Any]:
    """比較結果を JSON に書き出す辞書にまとめる。"""
    total = PageStats()
    for result in results:
        total = total + result.stats
    candidates = [c.to_dict() for r in results for c in r.candidates]
    return {
        "format_version": FORMAT_VERSION,
        "gemini_dir": str(gemini),
        "yomitoku_dir": str(yomitoku),
        "options": asdict(options),
        "page_count": len(results),
        "stats": asdict(total),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _score(truth: Path, gemini: Path, results: list[PageResult]) -> dict[str, Any]:
    """評価セットのページについて、誤読候補を正解の誤読と照らし合わせる。"""
    by_page = {r.page: r for r in results}
    pages: list[dict[str, Any]] = []
    scores: list[CrossOcrScore] = []
    for entry in load_manifest(truth):
        if entry.page not in by_page:
            raise FileNotFoundError(
                f"評価セットのページ {entry.page} が比較対象にありません: {gemini}"
            )
        misreads = find_gemini_misreads(
            read_truth(truth, entry.page), read_prediction(gemini, entry.page)
        )
        score = score_page(misreads, by_page[entry.page].candidates)
        scores.append(score)
        pages.append({"page": entry.page, "category": entry.category, **score.to_dict()})
    return {"pages": pages, "total": sum_scores(scores).to_dict()}


def _stats_table(output: dict[str, Any]) -> Table:
    table = Table(title=f"比較結果の内訳（{output['page_count']} ページ）")
    table.add_column("区分")
    table.add_column("件数", justify="right")
    labels = {
        "paragraphs": "yomitoku の段落",
        "paragraphs_in_figure": "  うち図の領域（除外）",
        "paragraphs_in_header": "  うちページ上部の帯・柱（除外）",
        "runs": "日本語の連続（run）",
        "runs_short": "  うち短すぎる（除外）",
        "runs_exact": "  うち Gemini と完全一致",
        "runs_different": "  うち Gemini と食い違い",
        "runs_unmatched": "  うち Gemini に対応箇所なし",
        "ops_misaligned": "対応の取り違えとして除いた食い違い",
    }
    for key, label in labels.items():
        table.add_row(label, str(output["stats"][key]))
    table.add_row("誤読候補", str(output["candidate_count"]))
    return table


def _candidate_table(results: list[PageResult]) -> Table:
    candidates = [c for r in results for c in r.candidates]
    table = Table(title=f"誤読候補（{len(candidates)} 件）")
    for column in ("ページ", "種別", "Gemini", "yomitoku"):
        table.add_column(column)
    for c in candidates:
        table.add_row(
            f"p.{c.page}",
            c.kind,
            f"{c.gemini_before[-6:]}[{c.gemini_text}]{c.gemini_after[:6]}",
            f"{c.yomitoku_before[-6:]}[{c.yomitoku_text}]{c.yomitoku_after[:6]}",
        )
    return table


def _score_table(scores: dict[str, Any]) -> Table:
    table = Table(title="評価セットでの適合率・再現率（Gemini の地の文の誤読単位）")
    for column in ("対象", "誤読候補", "うち誤読", "誤読", "検出した誤読", "適合率", "再現率"):
        table.add_column(column)
    rows = [(f"p.{p['page']}", p) for p in scores["pages"] if p["flagged"] or p["misread"]]
    for label, s in [*rows, ("全体", scores["total"])]:
        table.add_row(
            label,
            str(s["flagged"]),
            str(s["true_positive"]),
            str(s["misread"]),
            str(s["detected"]),
            f"{s['precision']:.3f}",
            f"{s['recall']:.3f}",
        )
    return table
