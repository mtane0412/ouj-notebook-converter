"""仕様: 評価セットでの LLM 判定の正誤と、修正適用後の指標の改善幅を測る。

判定の正誤の定義:
  - 正解ラベル: 候補が「真の誤読」か「誤検出」か。既知の人手修正（#28 の corrections.json）の
    置換前の文字列が候補の抜粋に含まれれば真の誤読、そうでなければ誤検出とする。
    抜粋に含まれない誤読や、人が原本で確かめた結果は手動ラベル（labels）で上書きする
  - 混同行列: 真の誤読を misread と判定 = TP、uncertain/correct と判定 = FN（見逃し）。
    誤検出を misread と判定 = FP、correct と判定 = TN。誤検出を uncertain と判定したものは
    uncertain_on_correct として別に数える（人の確認に回るだけで誤った修正は入らない）
  - 修正の正しさ: TP のうち、判定の置換を raw.md に適用した結果が、既知の修正を適用した結果と
    文字列として一致したもの

指標の改善幅は、評価セットのページに修正を適用して後処理（normalize_ocr_markdown）を通した
Markdown を評価セットの正解と比べて測る（ounc の変換と同じ順序）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ouj_notebook_converter.corrections import Correction, apply_corrections
from ouj_notebook_converter.evaluation.markdown_parts import Formula
from ouj_notebook_converter.evaluation.report import evaluate_dataset
from ouj_notebook_converter.judge.models import JudgeCandidate, Judgment
from ouj_notebook_converter.judge.policy import disambiguate
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import normalize_ocr_markdown

KatexChecker = Callable[[Sequence[Formula]], list[str | None]]


def derive_labels(
    candidates: Sequence[JudgeCandidate],
    known: Sequence[Correction],
    manual: Mapping[str, str],
    raw_pages: Mapping[int, str],
) -> dict[str, str]:
    """候補ごとの正解ラベル（misread / correct）を作る。手動ラベルが優先。"""
    labels: dict[str, str] = {}
    for candidate in candidates:
        if candidate.id in manual:
            labels[candidate.id] = manual[candidate.id]
        elif _known_for(candidate, known, raw_pages[candidate.page]):
            labels[candidate.id] = "misread"
        else:
            labels[candidate.id] = "correct"
    return labels


_FORMULA_SOURCES = frozenset({"equation_check", "katex_error"})


def _known_for(
    candidate: JudgeCandidate, known: Sequence[Correction], raw: str
) -> list[Correction]:
    """候補に対応する既知の修正を返す。

    数式の検出器の候補は、候補の数式（text）と置換前の範囲が重なる修正だけを対応とする
    （同じ抜粋の別の式の誤読を、この候補の期待値に混ぜないため）。
    それ以外の候補は、置換前の文字列が抜粋に含まれる修正を対応とする。
    """
    same_page = [c for c in known if c.page == candidate.page]
    if candidate.source in _FORMULA_SOURCES and candidate.text and candidate.text in raw:
        start = raw.index(candidate.text)
        end = start + len(candidate.text)
        return [
            c
            for c in same_page
            if c.before in raw
            and raw.index(c.before) < end
            and raw.index(c.before) + len(c.before) > start
        ]
    return [c for c in same_page if c.before in candidate.excerpt]


@dataclass
class AccuracyReport:
    """判定の正誤の集計。"""

    true_positive: int = 0
    fix_correct: int = 0
    fix_unverified: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    uncertain_on_correct: int = 0
    errors: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def misread_count(self) -> int:
        """真の誤読の候補数。"""
        return self.true_positive + self.false_negative

    @property
    def correct_count(self) -> int:
        """誤検出の候補数。"""
        return self.false_positive + self.true_negative + self.uncertain_on_correct

    def to_dict(self) -> dict[str, Any]:
        """JSON に書き出すための辞書（行を除く）。"""
        return {
            "misread_count": self.misread_count,
            "correct_count": self.correct_count,
            "true_positive": self.true_positive,
            "fix_correct": self.fix_correct,
            "fix_unverified": self.fix_unverified,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "uncertain_on_correct": self.uncertain_on_correct,
            "errors": self.errors,
        }


def _apply_fix(raw: str, judgment: Judgment) -> str | None:
    """判定の置換を raw に適用する。一意に適用できなければ None。"""
    if not judgment.before or raw.count(judgment.before) != 1:
        return None
    return raw.replace(judgment.before, judgment.after, 1)


def accuracy_report(
    candidates: Sequence[JudgeCandidate],
    judgments: Sequence[Judgment],
    labels: Mapping[str, str],
    raw_pages: Mapping[int, str],
    known: Sequence[Correction],
) -> AccuracyReport:
    """判定を正解ラベルと照らし合わせて集計する。"""
    by_id = {j.candidate_id: j for j in judgments}
    report = AccuracyReport()
    for candidate in candidates:
        judgment = by_id[candidate.id]
        truth = labels[candidate.id]
        fix_ok: bool | None = None
        if judgment.error:
            report.errors += 1
        if truth == "misread":
            if judgment.verdict == "misread":
                report.true_positive += 1
                raw = raw_pages[candidate.page]
                expected_for = _known_for(candidate, known, raw)
                if not expected_for:
                    report.fix_unverified += 1
                else:
                    expected = apply_corrections(raw, tuple(expected_for))
                    resolved = disambiguate(judgment, candidate.excerpt, raw)
                    fix_ok = _apply_fix(raw, resolved) == expected
                    report.fix_correct += int(fix_ok)
            else:
                report.false_negative += 1
        elif judgment.verdict == "misread":
            report.false_positive += 1
        elif judgment.verdict == "correct":
            report.true_negative += 1
        else:
            report.uncertain_on_correct += 1
        report.rows.append(
            {
                "candidate_id": candidate.id,
                "source": candidate.source,
                "truth": truth,
                "verdict": judgment.verdict,
                "confidence": judgment.confidence,
                "fix_correct": fix_ok,
            }
        )
    return report


def apply_to_pages(
    raw_pages: Mapping[int, str], corrections: Sequence[Correction]
) -> dict[int, str]:
    """raw.md のページに修正を記載順に適用する（ounc --corrections と同じ適用）。"""
    result = dict(raw_pages)
    for correction in corrections:
        result[correction.page] = apply_corrections(result[correction.page], (correction,))
    return result


def write_applied_pages(pages: Mapping[int, str], out_dir: Path) -> None:
    """後処理（normalize_ocr_markdown）を通した Markdown を page_NNNN.md として書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for page, text in pages.items():
        (out_dir / f"page_{page:04d}.md").write_text(normalize_ocr_markdown(text), encoding="utf-8")


def measure_scores(
    truth_dir: Path,
    raw_pages: Mapping[int, str],
    corrections: Sequence[Correction],
    work_dir: Path,
    *,
    katex_checker: KatexChecker,
) -> dict[str, float]:
    """修正を適用した評価セットのページを、正解と比べた指標（全体）を返す。"""
    write_applied_pages(apply_to_pages(raw_pages, corrections), work_dir)
    total = evaluate_dataset(truth_dir, work_dir, katex_checker=katex_checker).total
    return {
        "prose_cer": total.scores.cer.cer,
        "math_f1": total.scores.math.f1,
        "math_cer": total.scores.math_cer.cer,
        "heading_f1": total.scores.headings.f1,
        "katex_errors": float(total.katex_error_count),
    }
