"""仕様: 同一ページの複数回の OCR 結果を多数決で確定し、単独の実行との評価指標の差を測る（issue #27）。

多数決の方式:
  - 基準の実行（ページごとの medoid: 他の実行との差が最小の実行）に他の実行を整列し、
    食い違う箇所ごとに最も多くの実行が出した版を採用する。同数なら基準の実行の版を採用する
  - 地の文は文字単位（食い違う範囲ごと）、数式は数式 1 つ単位で投票する。
    数式の個数が実行間で違う場合も、食い違う範囲ごとに投票する
  - 見出しは地の文に含まれるため、独立には投票しない

評価指標（evaluation と同じ定義）:
  - 文字誤り率（CER）: 地の文の編集距離 ÷ 正解の文字数
  - 数式の一致率（F1）: 正規化した数式の多重集合としての適合率・再現率・F1
  - 数式の文字誤り率: 正規化した数式を出現順に連結した文字列の編集距離 ÷ 正解の文字数

注意事項:
  - 単独の実行も同じ関数（parts_of）で地の文・数式に分けてから指標を計算し、多数決の結果と同じ条件で比べる
  - 多数決は食い違い箇所だけを選び直す。全実行が同じ誤読をした箇所は直らない
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, TypeVar, cast

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    MATH_MARKER,
    extract_gemini_prose,
)
from ouj_notebook_converter.detectors.multi_run_variance.detect import align_runs
from ouj_notebook_converter.evaluation.markdown_parts import normalize_latex, split_markdown
from ouj_notebook_converter.evaluation.metrics import (
    CerScore,
    MatchScore,
    levenshtein,
    match_multiset,
)

SeqT = TypeVar("SeqT", bound=Sequence[Any])


@dataclass(frozen=True)
class PageParts:
    """評価の単位に分けた 1 ページ分の OCR 結果。

    Attributes:
        prose: 地の文（split_markdown().prose と同じ）。
        formulas: 出現順の、正規化した数式（日本語だけで空になるものを除く）。
    """

    prose: str
    formulas: tuple[str, ...]


@dataclass(frozen=True)
class VoteScores:
    """多数決（または単独の実行）の評価指標の元になる件数。"""

    cer: CerScore
    math: MatchScore
    math_cer: CerScore


def parts_of(markdown: str) -> PageParts:
    """Markdown を地の文と正規化した数式に分ける。"""
    return PageParts(
        prose=extract_gemini_prose(markdown).replace(MATH_MARKER, ""),
        formulas=_normalized_formulas(markdown),
    )


def vote_sequence(sequences: Sequence[SeqT], reference: int = 0) -> SeqT:
    """食い違う範囲ごとに最も多い版を採用した系列を返す。同数なら基準の実行の版を採用する。

    Args:
        sequences: 実行ごとの系列（文字列、または数式の列）。
        reference: 基準にする実行の番号。

    Returns:
        基準の実行の系列を、多数決で選んだ版に置き換えたもの。
    """
    # str と tuple のどちらでも同じ連結・スライスで扱うため Any として扱う
    ref: Any = sequences[reference]
    result: Any = ref[0:0]
    position = 0
    for cluster in align_runs(sequences, reference):
        variants = [sequences[run][s:e] for run, (s, e) in enumerate(cluster.spans)]
        start, end = cluster.spans[reference]
        result = result + ref[position:start] + _majority(variants, reference)
        position = end
    return cast(SeqT, result + ref[position:])


def pick_medoid(texts: Sequence[str]) -> int:
    """他の実行との差の合計が最小の実行（medoid）の番号を返す。同じなら番号が小さい方。"""
    totals = [
        sum(_distance(text, other) for j, other in enumerate(texts) if j != i)
        for i, text in enumerate(texts)
    ]
    return totals.index(min(totals))


def vote_page(run_markdowns: Sequence[str]) -> PageParts:
    """1 ページ分の複数回の OCR 結果を、地の文と数式のそれぞれで多数決して確定する。"""
    proses = [extract_gemini_prose(md) for md in run_markdowns]
    formulas = [_normalized_formulas(md) for md in run_markdowns]
    reference = pick_medoid(proses)
    return PageParts(
        prose=vote_sequence(proses, reference).replace(MATH_MARKER, ""),
        formulas=vote_sequence(formulas, reference),
    )


def medoid_page(run_markdowns: Sequence[str]) -> PageParts:
    """多数決をせず、medoid の実行をそのまま採用した場合の結果（比較用）。"""
    proses = [extract_gemini_prose(md) for md in run_markdowns]
    return parts_of(run_markdowns[pick_medoid(proses)])


def score_parts(truth: PageParts, pred: PageParts) -> VoteScores:
    """正解と評価対象を比べ、指標の元になる件数を返す。"""
    return VoteScores(
        cer=CerScore(edits=_edits(truth.prose, pred.prose), truth_length=len(truth.prose)),
        math=match_multiset(truth.formulas, pred.formulas),
        math_cer=CerScore(
            edits=_edits("".join(truth.formulas), "".join(pred.formulas)),
            truth_length=len("".join(truth.formulas)),
        ),
    )


def sum_vote_scores(scores: Sequence[VoteScores]) -> VoteScores:
    """複数ページの件数を合算する（マイクロ平均の元になる）。"""
    return VoteScores(
        cer=CerScore(
            edits=sum(s.cer.edits for s in scores),
            truth_length=sum(s.cer.truth_length for s in scores),
        ),
        math=MatchScore(
            matched=sum(s.math.matched for s in scores),
            truth_count=sum(s.math.truth_count for s in scores),
            pred_count=sum(s.math.pred_count for s in scores),
        ),
        math_cer=CerScore(
            edits=sum(s.math_cer.edits for s in scores),
            truth_length=sum(s.math_cer.truth_length for s in scores),
        ),
    )


def _majority(variants: Sequence[SeqT], reference: int) -> SeqT:
    """最も多い版を返す。最多が複数なら基準の実行の版、基準がその中に無ければ実行順で最初の版。"""
    counts = Counter(variants)
    top = max(counts.values())
    if counts[variants[reference]] == top:
        return variants[reference]
    return next(v for v in variants if counts[v] == top)


def _normalized_formulas(markdown: str) -> tuple[str, ...]:
    """数式を正規化する。日本語だけの数式は正規化後に空になるため除く（evaluation と同じ扱い）。"""
    normalized = (normalize_latex(f.tex) for f in split_markdown(markdown).formulas)
    return tuple(tex for tex in normalized if tex)


def _distance(a: str, b: str) -> int:
    """medoid を選ぶための、2 つの地の文の差の大きさ（食い違う範囲の長い方の文字数の合計）。"""
    opcodes = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in opcodes if tag != "equal")


@cache
def _edits(truth: str, pred: str) -> int:
    """編集距離。部分集合ごとの評価で同じ組を繰り返し計算するためキャッシュする。"""
    return levenshtein(truth, pred)
