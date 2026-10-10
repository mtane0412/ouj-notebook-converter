"""仕様: 判定結果を「自動適用」「人の確認待ち」「棄却」に振り分ける基準。

振り分けの規則（AutoPolicy で調整できる）:
  1. 判定に失敗した（error）／判断不能（uncertain）→ 人の確認待ち
  2. 正しい（correct）と判定
       - 確信度が low 以外 → 棄却（修正しない）
       - 確信度が low → 人の確認待ち（見逃しを避ける）
  3. 誤読（misread）と判定
       - 置換前の文字列が raw.md に 1 か所だけ現れない／置換が空・変化なし → 人の確認待ち
       - 確信度が min_confidence 未満 → 人の確認待ち
       - 検出器が auto_sources に含まれない → 人の確認待ち
       - 置換の変更文字数が max_edit_chars を超える（大きな書き換え）→ 人の確認待ち
       - それ以外 → 自動適用
自動適用の既定値は、評価セット（33 ページ）と 314 ページの測定結果から決めた（README 参照）。
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from ouj_notebook_converter.judge.models import (
    ALL_SOURCES,
    Confidence,
    JudgeCandidate,
    Judgment,
    Usage,
)

Decision = Literal["auto", "review", "reject"]
FixStatus = Literal["ok", "empty", "unchanged", "not_found", "ambiguous"]

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
DEFAULT_MAX_EDIT_CHARS = 8
# katex_error は「描画できない数式」で、誤読とは限らない（\cline など KaTeX が未対応の記法が原因）。
# LLM が数式を書き換えて「修正」を出しても原本の誤読とは限らないため、自動適用の対象にしない
DEFAULT_AUTO_SOURCES: frozenset[str] = frozenset(ALL_SOURCES) - {"katex_error"}


@dataclass(frozen=True)
class AutoPolicy:
    """自動適用の基準。

    Attributes:
        min_confidence: 自動適用に必要な最低の確信度。
        max_edit_chars: 自動適用できる置換の最大変更文字数（before と after の差分の大きい方の文字数の和）。
        auto_sources: 自動適用を許す検出器。
        min_agree: 自動適用に必要な、同じ修正に一致した判定（棄権を除く）の数。
    """

    min_confidence: Literal["low", "medium", "high"] = "high"
    max_edit_chars: int = DEFAULT_MAX_EDIT_CHARS
    auto_sources: frozenset[str] = DEFAULT_AUTO_SOURCES
    min_agree: int = 1


def check_fix(judgment: Judgment, raw: str) -> FixStatus:
    """判定の置換（before → after）が raw.md に適用できるかを調べる。"""
    if not judgment.before or not judgment.after:
        return "empty"
    if judgment.before == judgment.after:
        return "unchanged"
    count = raw.count(judgment.before)
    if count == 0:
        return "not_found"
    if count > 1:
        return "ambiguous"
    return "ok"


def edit_size(before: str, after: str) -> int:
    """before から after への変更文字数（差分の各ブロックで大きい方の文字数の和）。"""
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    )


# 置換前の文字列を一意にするために両側へ広げる文字数の上限
MAX_DISAMBIGUATION_CHARS = 40


def disambiguate(judgment: Judgment, excerpt: str, raw: str) -> Judgment:
    """ページ内で複数か所に一致する置換前の文字列を、抜粋の位置を手がかりに一意にして返す。

    LLM は抜粋の中の誤読箇所を指しているので、置換前の文字列が抜粋に 1 か所だけ現れるなら、
    その位置の前後の文字を 1 文字ずつ両側へ足し、ページ内で 1 か所に決まる長さにする。
    抜粋に 1 か所だけ現れない場合や、上限まで広げても一意にならない場合は、判定をそのまま返す。
    """
    before = judgment.before
    if not before or not judgment.after or raw.count(before) <= 1:
        return judgment
    if excerpt.count(before) != 1 or raw.count(excerpt) != 1:
        return judgment
    start = raw.index(excerpt) + excerpt.index(before)
    end = start + len(before)
    for grow in range(1, MAX_DISAMBIGUATION_CHARS + 1):
        left, right = max(0, start - grow), min(len(raw), end + grow)
        window = raw[left:right]
        if raw.count(window) == 1:
            head, tail = raw[left:start], raw[end:right]
            return replace(judgment, before=window, after=head + judgment.after + tail)
    return judgment


_FIX_MESSAGES: dict[str, str] = {
    "empty": "置換前または置換後が空です",
    "unchanged": "置換前と置換後が同じです",
    "not_found": "置換前の文字列が raw.md に見つかりません",
    "ambiguous": "置換前の文字列が raw.md の複数か所に一致します",
}


@dataclass(frozen=True)
class Consensus:
    """複数回の判定をまとめた結果。agree は同じ結論に一致した判定（棄権を除く）の数。"""

    judgment: Judgment
    agree: int


def _fix_result(judgment: Judgment, raw: str) -> str | None:
    """判定の置換を適用した結果。一意に適用できなければ None。"""
    if check_fix(judgment, raw) != "ok":
        return None
    return raw.replace(judgment.before, judgment.after, 1)


def combine(judgments: Sequence[Judgment], candidate: JudgeCandidate, raw: str) -> Consensus:
    """同じ候補に対する複数回の判定（切り出し画像・ページ全体・別モデルなど）を 1 つにまとめる。

    - 判定に失敗したもの・判断不能は棄権とし、一致数に数えない
    - 棄権を除いた全員が正しい（correct）なら正しい。確信度は最も低いものにする
    - 全員が誤読（misread）で、置換を適用した結果の文字列が同じなら誤読。確信度は最も低いものにする
    - 判定が割れた（誤読と正しいの混在・修正の食い違い）場合は、判断不能にする
    """
    resolved = [disambiguate(j, candidate.excerpt, raw) for j in judgments]
    voters = [j for j in resolved if not j.error and j.verdict != "uncertain"]
    merged_model = "+".join(dict.fromkeys(j.model for j in resolved))
    usage = Usage()
    for j in resolved:
        usage = usage + j.usage

    def lowest(items: Sequence[Judgment]) -> Confidence:
        return min((j.confidence for j in items), key=lambda c: _CONFIDENCE_RANK[c])

    if not voters:
        base = resolved[0]
        return Consensus(replace(base, model=merged_model, usage=usage), agree=0)
    verdicts = {j.verdict for j in voters}
    if verdicts == {"correct"}:
        base = voters[0]
        return Consensus(
            replace(base, model=merged_model, usage=usage, confidence=lowest(voters)),
            agree=len(voters),
        )
    if verdicts == {"misread"}:
        results = {_fix_result(j, raw) or f"{j.before}\0{j.after}" for j in voters}
        if len(results) == 1:
            base = voters[0]
            return Consensus(
                replace(base, model=merged_model, usage=usage, confidence=lowest(voters)),
                agree=len(voters),
            )
    summary = " / ".join(f"{j.model}: {j.verdict}" for j in voters)
    return Consensus(
        Judgment(
            candidate_id=voters[0].candidate_id,
            model=merged_model,
            verdict="uncertain",
            confidence="low",
            reason=f"判定が割れました（{summary}）",
            usage=usage,
        ),
        agree=0,
    )


def decide(
    candidate: JudgeCandidate,
    judgment: Judgment,
    raw: str,
    policy: AutoPolicy,
    agree: int = 1,
) -> tuple[Decision, str]:
    """判定結果の扱いと、その理由を返す。agree は同じ結論に一致した判定の数。"""
    if judgment.error:
        return "review", f"判定に失敗しました: {judgment.error}"
    if judgment.verdict == "uncertain":
        return "review", "LLM が原本の表記を確定できませんでした"
    if judgment.verdict == "correct":
        if judgment.confidence == "low":
            return "review", "正しいと判定されましたが確信度が低い"
        return "reject", "原本どおり（誤検出）と判定されました"

    fix = check_fix(judgment, raw)
    if fix != "ok":
        return "review", _FIX_MESSAGES[fix]
    if _CONFIDENCE_RANK[judgment.confidence] < _CONFIDENCE_RANK[policy.min_confidence]:
        return "review", f"確信度が {policy.min_confidence} 未満です（{judgment.confidence}）"
    if agree < policy.min_agree:
        return (
            "review",
            f"一致した判定が {agree} 件で、自動適用に必要な {policy.min_agree} 件に足りません",
        )
    if candidate.source not in policy.auto_sources:
        return "review", f"検出器 {candidate.source} は自動適用の対象外です"
    size = edit_size(judgment.before, judgment.after)
    if size > policy.max_edit_chars:
        return "review", f"変更が大きい（{size} 文字 > {policy.max_edit_chars}）"
    return "auto", "確信度が高く、小さな置換です"
