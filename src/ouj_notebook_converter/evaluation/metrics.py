"""仕様: 正解 Markdown と評価対象 Markdown を比較して精度指標を計算する。

指標:
  - 文字誤り率（CER）: 地の文（数式・Markdown 記法を除いた日本語部分）の編集距離 ÷ 正解の文字数
  - 数式の一致率: 正規化した数式の多重集合としての適合率・再現率・F1
  - 数式の文字誤り率: 正規化した数式を出現順に連結した文字列の編集距離 ÷ 正解の文字数。
    数式の分割・結合のしかたに左右されないため、一致率の補助指標とする
  - 見出し構造の一致率: (レベル, テキスト) の多重集合としての適合率・再現率・F1

注意事項:
  - 数式・見出しは出現順を問わず多重集合として比較する。OCR が数式を分割・結合した場合は不一致になる
  - 複数ページの集計は件数を合算してから比率を計算する（マイクロ平均）
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

from ouj_notebook_converter.evaluation.markdown_parts import (
    Formula,
    normalize_latex,
    split_markdown,
)


@dataclass(frozen=True)
class CerScore:
    """文字誤り率の元になる件数。

    Attributes:
        edits: 正解から評価対象への編集距離（挿入・削除・置換の回数）。
        truth_length: 正解の文字数。
    """

    edits: int
    truth_length: int

    @property
    def cer(self) -> float:
        """文字誤り率。正解が空の場合はゼロ除算を避けるため分母を 1 とする。"""
        return self.edits / max(self.truth_length, 1)


@dataclass(frozen=True)
class MatchScore:
    """多重集合としての一致件数。

    Attributes:
        matched: 一致した件数。
        truth_count: 正解の件数。
        pred_count: 評価対象の件数。
    """

    matched: int
    truth_count: int
    pred_count: int

    @property
    def precision(self) -> float:
        """適合率。評価対象が 0 件なら誤検出も無いため 1 とする。"""
        return self.matched / self.pred_count if self.pred_count else 1.0

    @property
    def recall(self) -> float:
        """再現率。正解が 0 件なら見落としも無いため 1 とする。"""
        return self.matched / self.truth_count if self.truth_count else 1.0

    @property
    def f1(self) -> float:
        """適合率と再現率の調和平均。どちらも 0 なら 0 とする。"""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass(frozen=True)
class PageScores:
    """1 ページ（または集計した複数ページ）の指標。"""

    cer: CerScore
    math: MatchScore
    math_cer: CerScore
    headings: MatchScore


def levenshtein(truth: str, pred: str) -> int:
    """2 つの文字列の編集距離（挿入・削除・置換をそれぞれ 1 回と数える）を返す。

    1 ページ数千文字程度を想定した O(len(truth) * len(pred)) の動的計画法で計算する。
    """
    if len(truth) < len(pred):
        truth, pred = pred, truth
    # previous[j]: truth の直前の行までと pred[:j] の編集距離
    previous = list(range(len(pred) + 1))
    for i, truth_char in enumerate(truth, start=1):
        current = [i]
        for j, pred_char in enumerate(pred, start=1):
            current.append(
                min(
                    previous[j] + 1,  # 削除
                    current[j - 1] + 1,  # 挿入
                    previous[j - 1] + (truth_char != pred_char),  # 置換（一致なら 0）
                )
            )
        previous = current
    return previous[-1]


def match_multiset(truth: Iterable[Hashable], pred: Iterable[Hashable]) -> MatchScore:
    """正解と評価対象を多重集合として比較し、一致件数を数える。

    同じ要素は、正解と評価対象のうち少ない方の個数までを一致とする。
    """
    truth_counter = Counter(truth)
    pred_counter = Counter(pred)
    return MatchScore(
        matched=sum((truth_counter & pred_counter).values()),
        truth_count=sum(truth_counter.values()),
        pred_count=sum(pred_counter.values()),
    )


def evaluate_page(truth_markdown: str, pred_markdown: str) -> PageScores:
    """1 ページ分の正解 Markdown と評価対象 Markdown を比較する。"""
    truth = split_markdown(truth_markdown)
    pred = split_markdown(pred_markdown)
    truth_math = _normalized_formulas(truth.formulas)
    pred_math = _normalized_formulas(pred.formulas)
    return PageScores(
        cer=_cer(truth.prose, pred.prose),
        math=match_multiset(truth_math, pred_math),
        math_cer=_cer("".join(truth_math), "".join(pred_math)),
        headings=match_multiset(truth.headings, pred.headings),
    )


def summarize(scores: Sequence[PageScores]) -> PageScores:
    """複数ページの指標の件数を合算する（マイクロ平均の元になる）。"""
    return PageScores(
        cer=_sum_cer([s.cer for s in scores]),
        math=_sum_match([s.math for s in scores]),
        math_cer=_sum_cer([s.math_cer for s in scores]),
        headings=_sum_match([s.headings for s in scores]),
    )


def _normalized_formulas(formulas: Sequence[Formula]) -> list[str]:
    """数式を正規化する。日本語だけの数式は正規化後に空になるため除く。"""
    return [tex for tex in (normalize_latex(f.tex) for f in formulas) if tex]


def _cer(truth: str, pred: str) -> CerScore:
    return CerScore(edits=levenshtein(truth, pred), truth_length=len(truth))


def _sum_cer(scores: Sequence[CerScore]) -> CerScore:
    return CerScore(
        edits=sum(s.edits for s in scores),
        truth_length=sum(s.truth_length for s in scores),
    )


def _sum_match(scores: Sequence[MatchScore]) -> MatchScore:
    return MatchScore(
        matched=sum(s.matched for s in scores),
        truth_count=sum(s.truth_count for s in scores),
        pred_count=sum(s.pred_count for s in scores),
    )
