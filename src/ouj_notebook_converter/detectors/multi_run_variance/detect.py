"""仕様: 同一ページの複数回の OCR 結果を整列し、実行間で食い違う箇所を不確実箇所として抽出する（issue #27）。

処理の流れ（1 ページ分）:
  1. 各実行の Markdown から、地の文（数式の位置に印を残したもの）と数式列（正規化した LaTeX）を取り出す
  2. 基準の実行に対して他の実行を文字（数式は 1 つ）単位で整列し、食い違う範囲を集める。
     異なる実行から得た範囲が重なる・接する場合は 1 つの範囲（クラスタ）にまとめ、実行ごとの文字列を並べる
  3. 地の文のクラスタを prose、数式のクラスタを formula の候補にする

候補を採用する条件: 基準の実行と文字列が異なる実行の数（dissent）が min_dissent 以上であること。

注意事項:
  - 候補は「実行間で揺れた箇所」であり、どれが正しいかは判定しない（後段の LLM 判定に委ねる）
  - 地の文は評価の地の文（split_markdown().prose）と同じ正規化をするため、Markdown 記法・空白・全角半角の違いでは揺れない
  - 数式は normalize_latex で表示が同じ表記の違い（\\dfrac と \\frac など）を除いて比べる
  - 実行間で数式の個数が違うだけの地の文の差（数式の印の有無）は、数式側で扱うため地の文の候補にしない
  - 全実行が同じ誤読をした箇所は揺れないため検出できない
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    MATH_LABEL,
    MATH_MARKER,
    extract_gemini_prose,
)
from ouj_notebook_converter.evaluation.markdown_parts import normalize_latex, split_markdown

# 候補に付ける前後の文脈の文字数
DEFAULT_CONTEXT_CHARS = 15

Granularity = Literal["prose", "formula"]
SeqT = TypeVar("SeqT", bound=Sequence[Any])


@dataclass(frozen=True)
class DetectOptions:
    """検出の設定。

    Attributes:
        min_dissent: 基準の実行と異なる実行がこの数以上ある食い違いだけを候補にする。
        context_chars: 候補に付ける前後の文脈の文字数。
    """

    min_dissent: int = 1
    context_chars: int = DEFAULT_CONTEXT_CHARS


@dataclass(frozen=True)
class Cluster:
    """複数の実行で食い違う 1 か所。

    Attributes:
        spans: 実行ごとの範囲 [開始, 終了)。実行自身の系列での位置。空の範囲は挿入位置を表す。
    """

    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VarianceCandidate:
    """実行間で食い違う 1 か所（不確実箇所の候補）。

    Attributes:
        page: ページ番号（1 始まり）。
        granularity: prose（地の文）または formula（数式）。
        reference_run: 基準にした実行の番号（実行の一覧での 0 始まりの位置）。
        texts: 実行ごとの文字列（実行の一覧と同じ順）。数式は正規化前の LaTeX を改行でつなぐ。
            当該箇所が空（その実行に対応する文字・数式が無い）なら空文字列。
        context_before: 基準の実行の地の文での直前の文脈。数式のあった位置は「〔数式〕」と書く。
        context_after: 基準の実行の地の文での直後の文脈。
        dissent: 基準の実行と文字列が異なる実行の数。
        run_count: 比較した実行の数。
        reference_span: 基準の実行の系列での範囲。prose は地の文の文字位置、formula は
            空でない数式だけを数えた通し番号。空の範囲は挿入位置を表す。
        formula_indices: formula のみ。範囲に含まれる基準の実行の数式の番号（全数式を出現順に数えた 0 始まり）。
            基準の実行に数式が無い（他の実行にだけ数式がある）場合は空。
    """

    page: int
    granularity: Granularity
    reference_run: int
    texts: tuple[str, ...]
    context_before: str
    context_after: str
    dissent: int
    run_count: int
    reference_span: tuple[int, int]
    formula_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """後段の LLM 判定への入力として JSON に書き出す辞書を返す。"""
        return {
            "page": self.page,
            "granularity": self.granularity,
            "reference_run": self.reference_run,
            "run_count": self.run_count,
            "dissent": self.dissent,
            "texts": list(self.texts),
            "context_before": self.context_before,
            "context_after": self.context_after,
            "reference_span": list(self.reference_span),
            "formula_indices": list(self.formula_indices),
        }


def align_runs(sequences: Sequence[SeqT], reference: int = 0) -> list[Cluster]:
    """基準の実行に対して他の実行を整列し、食い違う範囲を実行ごとの範囲つきでまとめる。

    Args:
        sequences: 実行ごとの系列（文字列、または数式の列）。
        reference: 基準にする実行の番号。

    Returns:
        食い違いのクラスタ（基準の実行での位置の昇順）。
    """
    ref = sequences[reference]
    opcodes_by_run: dict[int, Sequence[tuple[str, int, int, int, int]]] = {}
    pieces: list[tuple[int, int]] = []
    for run, sequence in enumerate(sequences):
        if run == reference:
            continue
        run_opcodes = difflib.SequenceMatcher(None, ref, sequence, autojunk=False).get_opcodes()
        opcodes_by_run[run] = run_opcodes
        pieces.extend((i1, i2) for tag, i1, i2, _, _ in run_opcodes if tag != "equal")

    clusters: list[Cluster] = []
    for start, end in _merge_ranges(pieces):
        spans: list[tuple[int, int]] = []
        for run in range(len(sequences)):
            if run == reference:
                spans.append((start, end))
            else:
                opcodes = opcodes_by_run[run]
                spans.append(
                    (_map_position(opcodes, start, "start"), _map_position(opcodes, end, "end"))
                )
        clusters.append(Cluster(spans=tuple(spans)))
    return clusters


def detect_page(
    page: int,
    run_markdowns: Sequence[str],
    reference: int = 0,
    options: DetectOptions | None = None,
) -> list[VarianceCandidate]:
    """1 ページ分の複数回の OCR 結果から、不確実箇所の候補を抽出する。

    Args:
        page: ページ番号（1 始まり）。
        run_markdowns: 実行ごとの Markdown。
        reference: 基準にする実行の番号。
        options: 検出の設定。

    Returns:
        候補（地の文、数式の順）。
    """
    options = options or DetectOptions()
    prose = _prose_candidates(page, run_markdowns, reference, options)
    formulas = _formula_candidates(page, run_markdowns, reference, options)
    return [*prose, *formulas]


def _prose_candidates(
    page: int, run_markdowns: Sequence[str], reference: int, options: DetectOptions
) -> list[VarianceCandidate]:
    proses = [extract_gemini_prose(md) for md in run_markdowns]
    ref_prose = proses[reference]
    candidates: list[VarianceCandidate] = []
    for cluster in align_runs(proses, reference):
        variants = [proses[run][s:e] for run, (s, e) in enumerate(cluster.spans)]
        # 数式の個数の違いだけの食い違いは、数式側の候補で扱う
        if len({v.replace(MATH_MARKER, "") for v in variants}) == 1:
            continue
        dissent = sum(v != variants[reference] for v in variants)
        if dissent < options.min_dissent:
            continue
        start, end = cluster.spans[reference]
        candidates.append(
            VarianceCandidate(
                page=page,
                granularity="prose",
                reference_run=reference,
                texts=tuple(v.replace(MATH_MARKER, MATH_LABEL) for v in variants),
                context_before=_label(ref_prose[max(0, start - options.context_chars) : start]),
                context_after=_label(ref_prose[end : end + options.context_chars]),
                dissent=dissent,
                run_count=len(run_markdowns),
                reference_span=(start, end),
            )
        )
    return candidates


def _formula_candidates(
    page: int, run_markdowns: Sequence[str], reference: int, options: DetectOptions
) -> list[VarianceCandidate]:
    # 実行ごとに (元の番号, 元の LaTeX, 正規化した LaTeX) を並べる。日本語だけの数式は正規化で空になるので除く
    items: list[list[tuple[int, str, str]]] = []
    for md in run_markdowns:
        run_items = []
        for index, formula in enumerate(split_markdown(md).formulas):
            normalized = normalize_latex(formula.tex)
            if normalized:
                run_items.append((index, formula.tex, normalized))
        items.append(run_items)
    sequences = [tuple(norm for _, _, norm in run_items) for run_items in items]
    ref_prose = extract_gemini_prose(run_markdowns[reference])
    marker_positions = [i for i, ch in enumerate(ref_prose) if ch == MATH_MARKER]

    candidates: list[VarianceCandidate] = []
    for cluster in align_runs(sequences, reference):
        variants = [sequences[run][s:e] for run, (s, e) in enumerate(cluster.spans)]
        dissent = sum(v != variants[reference] for v in variants)
        if dissent < options.min_dissent:
            continue
        start, end = cluster.spans[reference]
        ref_items = items[reference]
        formula_indices = tuple(ref_items[k][0] for k in range(start, end))
        texts = tuple(
            "\n".join(raw for _, raw, _ in items[run][s:e])
            for run, (s, e) in enumerate(cluster.spans)
        )
        before, after = _formula_context(
            ref_prose, marker_positions, _anchor_formula_index(ref_items, start, end), options
        )
        candidates.append(
            VarianceCandidate(
                page=page,
                granularity="formula",
                reference_run=reference,
                texts=texts,
                context_before=before,
                context_after=after,
                dissent=dissent,
                run_count=len(run_markdowns),
                reference_span=(start, end),
                formula_indices=formula_indices,
            )
        )
    return candidates


def _anchor_formula_index(
    ref_items: Sequence[tuple[int, str, str]], start: int, end: int
) -> int | None:
    """文脈を取る基準にする、基準の実行の数式の番号（全数式での番号）を返す。

    範囲に数式があれば先頭の数式、挿入位置なら直後（無ければ直前）の数式。数式が 1 つも無ければ None。
    """
    if start < end:
        return ref_items[start][0]
    if start < len(ref_items):
        return ref_items[start][0]
    if ref_items:
        return ref_items[-1][0]
    return None


def _formula_context(
    ref_prose: str, marker_positions: Sequence[int], index: int | None, options: DetectOptions
) -> tuple[str, str]:
    """n 番目の数式の印の前後の地の文を返す。"""
    if index is None or index >= len(marker_positions):
        return "", ""
    position = marker_positions[index]
    before = ref_prose[max(0, position - options.context_chars) : position]
    after = ref_prose[position + 1 : position + 1 + options.context_chars]
    return _label(before), _label(after)


def _label(text: str) -> str:
    """数式の印を「〔数式〕」に置き換える。"""
    return text.replace(MATH_MARKER, MATH_LABEL)


def _merge_ranges(pieces: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """重なる・接する範囲（空の範囲を含む）を 1 つにまとめる。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(pieces):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _map_position(
    opcodes: Sequence[tuple[str, int, int, int, int]], position: int, side: Literal["start", "end"]
) -> int:
    """基準の実行での位置を、整列した相手の実行での位置に写す。

    食い違い（挿入を含む）の境界では、範囲の開始なら食い違いの手前、終了なら食い違いの後ろに写す。
    """
    mapped: list[int] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if not i1 <= position <= i2:
            continue
        if tag == "equal":
            mapped.append(j1 + (position - i1))
        elif i1 < position < i2:
            # 食い違いの内側は範囲の端にならない（食い違いは必ずクラスタに丸ごと含まれる）
            raise ValueError(f"食い違いの内側の位置は写せません: {position}")
        else:
            if position == i1:
                mapped.append(j1)
            if position == i2:
                mapped.append(j2)
    return min(mapped) if side == "start" else max(mapped)
