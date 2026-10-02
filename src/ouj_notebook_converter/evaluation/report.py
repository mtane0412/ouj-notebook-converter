"""仕様: 評価セット全体を評価し、ページ別・層（カテゴリ）別・全体の指標をまとめる。

KaTeX 検査は外部プロセスを要するため、検査関数を引数で受け取る（テストではフェイクに差し替える）。
KaTeX エラーは評価対象の出力についてのみ数える。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ouj_notebook_converter.evaluation.dataset import load_manifest, read_prediction, read_truth
from ouj_notebook_converter.evaluation.markdown_parts import Formula, split_markdown
from ouj_notebook_converter.evaluation.metrics import (
    MatchScore,
    PageScores,
    evaluate_page,
    summarize,
)

KatexChecker = Callable[[Sequence[Formula]], list[str | None]]


@dataclass(frozen=True)
class PageReport:
    """1 ページの評価結果。

    Attributes:
        katex_errors: 評価対象の出力中で KaTeX が描画できなかった数式のエラーメッセージ。
    """

    page: int
    category: str
    scores: PageScores
    katex_errors: tuple[str, ...]


@dataclass(frozen=True)
class SummaryReport:
    """複数ページを集計した評価結果。"""

    page_count: int
    scores: PageScores
    katex_error_count: int


@dataclass(frozen=True)
class DatasetReport:
    """評価セット全体の評価結果。"""

    pages: tuple[PageReport, ...]
    by_category: dict[str, SummaryReport]
    total: SummaryReport

    def to_dict(self) -> dict[str, Any]:
        """JSON 出力用に、計算済みの指標（CER・F1 など）を含む辞書へ変換する。"""
        return {
            "pages": [
                {
                    "page": row.page,
                    "category": row.category,
                    **_metrics_dict(row.scores, katex_errors=len(row.katex_errors)),
                    "katex_error_messages": list(row.katex_errors),
                }
                for row in self.pages
            ],
            "by_category": {
                category: _summary_dict(summary) for category, summary in self.by_category.items()
            },
            "total": _summary_dict(self.total),
        }


def evaluate_dataset(
    truth_dir: Path, pred_dir: Path, *, katex_checker: KatexChecker
) -> DatasetReport:
    """評価セットの全ページについて正解と評価対象を比較する。

    Args:
        truth_dir: 評価セット（manifest.json と正解 page_NNNN.md）のディレクトリ。
        pred_dir: 評価対象の出力（--no-combine 出力またはページキャッシュ）のディレクトリ。
        katex_checker: 数式ごとの KaTeX エラーメッセージ（描画できれば None）を返す関数。

    Raises:
        FileNotFoundError / ValueError: 評価セットまたは評価対象のページが読み込めない場合。
    """
    entries = load_manifest(truth_dir)
    # 全ページを先に読み込み、欠けているページがあれば評価を始める前に失敗させる
    pairs = [(read_truth(truth_dir, e.page), read_prediction(pred_dir, e.page)) for e in entries]

    # KaTeX 検査は外部プロセスの起動を 1 回に抑えるため、全ページの数式をまとめて渡す
    pred_formulas = [split_markdown(pred).formulas for _, pred in pairs]
    katex_results = katex_checker([f for formulas in pred_formulas for f in formulas])

    rows: list[PageReport] = []
    offset = 0
    for entry, (truth, pred), formulas in zip(entries, pairs, pred_formulas, strict=True):
        page_results = katex_results[offset : offset + len(formulas)]
        offset += len(formulas)
        rows.append(
            PageReport(
                page=entry.page,
                category=entry.category,
                scores=evaluate_page(truth, pred),
                katex_errors=tuple(r for r in page_results if r is not None),
            )
        )

    categories = dict.fromkeys(row.category for row in rows)
    return DatasetReport(
        pages=tuple(rows),
        by_category={c: _summarize_rows([r for r in rows if r.category == c]) for c in categories},
        total=_summarize_rows(rows),
    )


def _summarize_rows(rows: Sequence[PageReport]) -> SummaryReport:
    return SummaryReport(
        page_count=len(rows),
        scores=summarize([row.scores for row in rows]),
        katex_error_count=sum(len(row.katex_errors) for row in rows),
    )


def _summary_dict(summary: SummaryReport) -> dict[str, Any]:
    return {
        "page_count": summary.page_count,
        **_metrics_dict(summary.scores, katex_errors=summary.katex_error_count),
    }


def _metrics_dict(scores: PageScores, *, katex_errors: int) -> dict[str, Any]:
    return {
        "cer": scores.cer.cer,
        "cer_edits": scores.cer.edits,
        "cer_truth_length": scores.cer.truth_length,
        **_match_dict("math", scores.math),
        **_match_dict("heading", scores.headings),
        "katex_errors": katex_errors,
    }


def _match_dict(prefix: str, score: MatchScore) -> dict[str, Any]:
    return {
        f"{prefix}_precision": score.precision,
        f"{prefix}_recall": score.recall,
        f"{prefix}_f1": score.f1,
        f"{prefix}_matched": score.matched,
        f"{prefix}_truth_count": score.truth_count,
        f"{prefix}_pred_count": score.pred_count,
    }
