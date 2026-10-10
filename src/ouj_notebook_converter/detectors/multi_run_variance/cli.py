"""仕様: 同一ページの複数回の OCR 結果の揺れから不確実箇所を抽出する検出器（issue #27）の CLI。

使い方:
  uv run python -m ouj_notebook_converter.detectors.multi_run_variance \\
      <実行1の出力> <実行2の出力> <実行3の出力> ... \\
      [--truth <評価セット>] [--json 結果.json]

- 各引数は 1 回分の OCR の出力（--no-combine の出力、またはページキャッシュ）。2 つ以上指定する
- 最初の実行を基準にして、実行間で食い違う箇所（地の文・数式）を不確実箇所の候補として表示する
- --truth を指定すると、評価セットのページについて次を計算する
    - 候補と誤読の重なりから求めた適合率・再現率（各実行を順に基準にして合算。しきい値 dissent ごと）
    - 多数決で確定した結果と単独の実行の評価指標（CER・数式F1・数式CER）
  --subset-size を指定すると、実行の部分集合（組み合わせ）ごとに計算して合算する
  （例: 5 回の実行から 3 回を選ぶ全 10 通りの平均）。既定は全実行を使う場合のみ
- --json は後段の LLM 判定（issue #24）への入力にもなる（形式は README を参照）
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.detectors.multi_run_variance.detect import (
    DEFAULT_CONTEXT_CHARS,
    DetectOptions,
    VarianceCandidate,
    detect_page,
)
from ouj_notebook_converter.detectors.multi_run_variance.scoring import (
    DetectionScores,
    score_candidates,
    sum_detection_scores,
)
from ouj_notebook_converter.detectors.multi_run_variance.voting import (
    VoteScores,
    medoid_page,
    parts_of,
    score_parts,
    sum_vote_scores,
    vote_page,
)
from ouj_notebook_converter.evaluation.dataset import (
    list_prediction_pages,
    load_manifest,
    read_prediction,
    read_truth,
)

# JSON の形式のバージョン（後段の LLM 判定が形式の変更を検知するため）
FORMAT_VERSION = 1
# 比較に必要な最小の実行数
MIN_RUNS = 2

app = typer.Typer(add_completion=False)


@app.command()
def multi_run_variance(
    run_dirs: Annotated[
        list[Path], typer.Argument(help="1 回分の OCR の出力（2 つ以上。最初のものを基準にする）")
    ],
    truth: Annotated[
        Path | None,
        typer.Option(
            "--truth", help="評価セット（指定すると適合率・再現率と多数決の効果を計算する）"
        ),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="検出結果を書き出す JSON ファイル")
    ] = None,
    min_dissent: Annotated[
        int,
        typer.Option(
            "--min-dissent", min=1, help="基準と異なる実行がこの数以上の食い違いだけ候補にする"
        ),
    ] = 1,
    context_chars: Annotated[
        int, typer.Option("--context-chars", min=0, help="候補に付ける前後の文脈の文字数")
    ] = DEFAULT_CONTEXT_CHARS,
    subset_size: Annotated[
        list[int] | None,
        typer.Option("--subset-size", help="評価に使う実行の数（複数指定可。既定は全実行）"),
    ] = None,
) -> None:
    """複数回の OCR 結果の食い違いを不確実箇所として表示する。"""
    console = Console()
    try:
        if len(run_dirs) < MIN_RUNS:
            raise ValueError(f"実行の出力は {MIN_RUNS} つ以上指定してください: {len(run_dirs)} つ")
        sizes = subset_size or [len(run_dirs)]
        for size in sizes:
            if not MIN_RUNS <= size <= len(run_dirs):
                raise ValueError(
                    f"--subset-size は {MIN_RUNS} 以上 {len(run_dirs)} 以下で指定してください: {size}"
                )
        pages = _target_pages(run_dirs, truth)
        runs_by_page = {p: [read_prediction(d, p) for d in run_dirs] for p in pages}
        options = DetectOptions(min_dissent=min_dissent, context_chars=context_chars)
        candidates = [c for p in pages for c in detect_page(p, runs_by_page[p], 0, options)]
        output: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "run_dirs": [str(d) for d in run_dirs],
            "options": {"min_dissent": min_dissent, "context_chars": context_chars},
            "page_count": len(pages),
            "candidate_count": len(candidates),
            "candidates": [c.to_dict() for c in candidates],
        }
        if truth is not None:
            output["scores"] = _score(truth, pages, runs_by_page, sizes)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    console.print(_candidate_table(candidates, len(run_dirs)))
    if "scores" in output:
        console.print(_detection_table(output["scores"]["detection"]))
        console.print(_vote_table(output["scores"]["vote"]))
    if json_path is not None:
        try:
            json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(code=1) from e
        console.print(f"検出結果を書き出しました: {json_path}")


def _target_pages(run_dirs: Sequence[Path], truth: Path | None) -> list[int]:
    """比較するページ。評価セットがあればそのページ、無ければ最初の実行にある全ページ。"""
    if truth is not None:
        return [entry.page for entry in load_manifest(truth)]
    return list_prediction_pages(run_dirs[0])


def _score(
    truth: Path, pages: Sequence[int], runs_by_page: dict[int, list[str]], sizes: Sequence[int]
) -> dict[str, Any]:
    """部分集合の大きさごとに、検出の適合率・再現率と多数決の効果を計算する。"""
    run_count = len(next(iter(runs_by_page.values())))
    truths = {p: read_truth(truth, p) for p in pages}
    detection: dict[str, Any] = {}
    vote: dict[str, Any] = {}
    for size in sizes:
        by_dissent: dict[int, list[DetectionScores]] = {d: [] for d in range(1, size)}
        singles: list[VoteScores] = []
        medoids: list[VoteScores] = []
        voted: list[VoteScores] = []
        for combo in itertools.combinations(range(run_count), size):
            for page in pages:
                runs = [runs_by_page[page][i] for i in combo]
                # しきい値を変えて数え直せるよう、dissent 1 で検出してから絞り込む
                candidates = [
                    detect_page(page, runs, r, DetectOptions(min_dissent=1)) for r in range(size)
                ]
                for dissent, scores in by_dissent.items():
                    scores.append(score_candidates(truths[page], runs, candidates, dissent))
                truth_parts = parts_of(truths[page])
                singles.extend(score_parts(truth_parts, parts_of(md)) for md in runs)
                medoids.append(score_parts(truth_parts, medoid_page(runs)))
                voted.append(score_parts(truth_parts, vote_page(runs)))
        detection[str(size)] = {
            str(d): sum_detection_scores(s).to_dict() for d, s in by_dissent.items()
        }
        vote[str(size)] = {
            "single_mean": _vote_dict(sum_vote_scores(singles)),
            "medoid": _vote_dict(sum_vote_scores(medoids)),
            "voted": _vote_dict(sum_vote_scores(voted)),
        }
    return {"detection": detection, "vote": vote}


def _vote_dict(score: VoteScores) -> dict[str, float | int]:
    """指標を JSON に書き出す辞書にする。"""
    return {
        "cer": score.cer.cer,
        "cer_edits": score.cer.edits,
        "math_f1": score.math.f1,
        "math_precision": score.math.precision,
        "math_recall": score.math.recall,
        "math_cer": score.math_cer.cer,
    }


def _candidate_table(candidates: Sequence[VarianceCandidate], run_count: int) -> Table:
    table = Table(title=f"不確実箇所の候補（{len(candidates)} 件、{run_count} 回の実行）")
    for column in ("ページ", "種別", "食い違う実行", "文脈", "各実行の文字列"):
        table.add_column(column)
    for c in candidates:
        table.add_row(
            f"p.{c.page}",
            c.granularity,
            f"{c.dissent}/{c.run_count - 1}",
            f"{c.context_before[-6:]}…{c.context_after[:6]}",
            " | ".join(t or "∅" for t in c.texts),
        )
    return table


def _detection_table(detection: dict[str, Any]) -> Table:
    table = Table(title="揺れと誤読の重なり（各実行を順に基準にして合算）")
    for column in ("実行数", "dissent≧", "種別", "候補", "うち誤読", "誤読", "適合率", "再現率"):
        table.add_column(column)
    for size, by_dissent in detection.items():
        for dissent, scores in by_dissent.items():
            for granularity, label in (("prose", "地の文"), ("formula", "数式")):
                s = scores[granularity]
                table.add_row(
                    size,
                    dissent,
                    label,
                    str(s["flagged"]),
                    str(s["true_positive"]),
                    str(s["misread"]),
                    f"{s['precision']:.3f}",
                    f"{s['recall']:.3f}",
                )
    return table


def _vote_table(vote: dict[str, Any]) -> Table:
    table = Table(title="多数決の効果（単独の実行・medoid・多数決）")
    for column in ("実行数", "方式", "CER", "数式F1", "数式CER"):
        table.add_column(column)
    labels = {"single_mean": "単独の実行（平均）", "medoid": "medoid の実行", "voted": "多数決"}
    for size, methods in vote.items():
        for key, label in labels.items():
            m = methods[key]
            table.add_row(
                size, label, f"{m['cer']:.2%}", f"{m['math_f1']:.3f}", f"{m['math_cer']:.2%}"
            )
    return table
