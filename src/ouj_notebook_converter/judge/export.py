"""仕様: 判定結果を修正ファイル（issue #28、JSON version 1）に書き出す。

出力（出力ディレクトリ）:
  - corrections.json          自動適用する修正。`ounc --corrections` にそのまま渡せる
  - corrections_pending.json  人の確認待ちの修正（置換は提案できているが、自動適用の基準を満たさない）。
                              確認して採用するものを corrections.json に移す。reviewer は空
  - decisions.json            全候補の振り分け結果と理由（棄却・判断不能を含む）

同じページ内の修正は、raw.md に記載順で順に適用できなければならない（#28 の規則）。
そのため、候補の順に適用を試し、前の修正と重なって置換前の文字列が壊れるものは確認待ちに回す。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ouj_notebook_converter.corrections import (
    SUPPORTED_VERSION,
    Correction,
    CorrectionError,
    apply_corrections,
)
from ouj_notebook_converter.judge.models import JudgeCandidate, Judgment
from ouj_notebook_converter.judge.policy import AutoPolicy, check_fix, combine, decide

CORRECTIONS_FILE = "corrections.json"
PENDING_FILE = "corrections_pending.json"
DECISIONS_FILE = "decisions.json"


@dataclass(frozen=True)
class Exports:
    """書き出す内容。decisions は全候補の振り分け結果（dict のリスト）。"""

    auto: tuple[Correction, ...]
    pending: tuple[Correction, ...]
    decisions: tuple[dict[str, Any], ...]


def _correction(
    candidate: JudgeCandidate, judgment: Judgment, *, reviewer: str, date: str, note: str = ""
) -> Correction:
    return Correction(
        page=candidate.page,
        before=judgment.before,
        after=judgment.after,
        reason=(
            f"{note}LLM判定({judgment.model}, 確信度 {judgment.confidence}, {candidate.id}): "
            f"{judgment.reason}"
        ),
        reviewer=reviewer,
        date=date,
    )


def _try_apply(texts: dict[int, str], correction: Correction) -> bool:
    """texts[page] に修正を適用できれば適用して True、できなければ変更せず False。"""
    try:
        texts[correction.page] = apply_corrections(texts[correction.page], (correction,))
    except CorrectionError:
        return False
    return True


def build_exports(
    candidates: Sequence[JudgeCandidate],
    runs: Sequence[Sequence[Judgment]],
    raw_pages: Mapping[int, str],
    policy: AutoPolicy,
    *,
    date: str,
) -> Exports:
    """判定結果を振り分け、自動適用・確認待ちの修正と振り分け結果を作る。

    Raises:
        ValueError: 候補に対応する判定結果が無い場合。
    """
    by_run = [{j.candidate_id: j for j in judgments} for judgments in runs]
    for index, by_id in enumerate(by_run, 1):
        missing = [c.id for c in candidates if c.id not in by_id]
        if missing:
            raise ValueError(f"{index} 番目の判定結果が無い候補があります: {missing[:5]}")

    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        raw = raw_pages[candidate.page]
        # 複数回の判定を 1 つにまとめる（置換前が曖昧なら、抜粋の位置を手がかりに一意な長さへ広げる）
        consensus = combine([by_id[candidate.id] for by_id in by_run], candidate, raw)
        judgment = consensus.judgment
        decision, reason = decide(candidate, judgment, raw, policy, consensus.agree)
        entries.append(
            {
                "candidate": candidate,
                "judgment": judgment,
                "agree": consensus.agree,
                "decision": decision,
                "reason": reason,
            }
        )

    # 自動適用 → 確認待ちの順に、記載順の適用が成り立つか確かめる
    texts = dict(raw_pages)
    auto: list[Correction] = []
    for entry in entries:
        if entry["decision"] != "auto":
            continue
        correction = _correction(
            entry["candidate"],
            entry["judgment"],
            reviewer=f"llm-judge:{entry['judgment'].model}",
            date=date,
        )
        if _try_apply(texts, correction):
            auto.append(correction)
        elif (
            correction.before not in texts[correction.page]
            and correction.after in texts[correction.page]
        ):
            # 別の候補（同じ誤読を見つけた他の検出器など）の修正で、既に直っている
            entry["reason"] = (
                "同じ箇所の修正が他の候補で適用済みです（置換後の文字列が既にあります）"
            )
        else:
            entry["decision"] = "review"
            entry["reason"] = "他の修正と重なるため、続けて適用できません（重複）"

    pending: list[Correction] = []
    for entry in entries:
        judgment = entry["judgment"]
        if entry["decision"] != "review" or judgment.verdict != "misread":
            continue
        if check_fix(judgment, raw_pages[entry["candidate"].page]) != "ok":
            continue
        correction = _correction(
            entry["candidate"], judgment, reviewer="", date=date, note="[要確認] "
        )
        if _try_apply(texts, correction):
            pending.append(correction)
        else:
            entry["reason"] += "（他の修正と重なるため、確認待ちの修正ファイルには含めません）"

    decisions = tuple(
        {
            "candidate_id": e["candidate"].id,
            "page": e["candidate"].page,
            "source": e["candidate"].source,
            "decision": e["decision"],
            "reason": e["reason"],
            "agree": e["agree"],
            "verdict": e["judgment"].verdict,
            "confidence": e["judgment"].confidence,
            "before": e["judgment"].before,
            "after": e["judgment"].after,
        }
        for e in entries
    )
    return Exports(auto=tuple(auto), pending=tuple(pending), decisions=decisions)


def _corrections_json(corrections: Sequence[Correction]) -> str:
    items = []
    for c in corrections:
        item: dict[str, Any] = {"page": c.page, "before": c.before, "after": c.after}
        for key in ("reason", "reviewer", "date"):
            if getattr(c, key):
                item[key] = getattr(c, key)
        items.append(item)
    return json.dumps(
        {"version": SUPPORTED_VERSION, "corrections": items}, ensure_ascii=False, indent=2
    )


def write_exports(exports: Exports, out_dir: Path) -> None:
    """corrections.json・corrections_pending.json・decisions.json を out_dir に書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CORRECTIONS_FILE).write_text(_corrections_json(exports.auto), encoding="utf-8")
    (out_dir / PENDING_FILE).write_text(_corrections_json(exports.pending), encoding="utf-8")
    (out_dir / DECISIONS_FILE).write_text(
        json.dumps(list(exports.decisions), ensure_ascii=False, indent=2), encoding="utf-8"
    )
