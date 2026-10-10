"""仕様: judge.evaluate のユニットテスト。

評価セットでの判定の正誤（混同行列・修正の正しさ）と、
判定結果を修正ファイルとして適用した後の指標の改善幅の計算をテストする。
"""

import json
from pathlib import Path
from typing import Any

from ouj_notebook_converter.corrections import Correction
from ouj_notebook_converter.judge.evaluate import (
    accuracy_report,
    apply_to_pages,
    derive_labels,
    measure_scores,
    write_applied_pages,
)
from ouj_notebook_converter.judge.models import JudgeCandidate, Judgment

_RAW = {
    60: "数の並びが無制限に繰り返される。\n別の行。\n",
    70: "著者は服部正博です。\n",
}
_KNOWN = (
    Correction(page=60, before="無制限に", after="無限に"),
    Correction(page=70, before="服部", after="隈部"),
)


def _candidate(candidate_id: str, page: int, excerpt: str) -> JudgeCandidate:
    return JudgeCandidate(
        id=candidate_id,
        page=page,
        source="cross_ocr_diff",
        kind="replace",
        text="",
        hint="",
        excerpt=excerpt,
        excerpt_scope="lines",
        line_index=0,
    )


def _judgment(
    candidate_id: str, verdict: str, before: str = "", after: str = "", **kw: Any
) -> Judgment:
    return Judgment(
        candidate_id=candidate_id,
        model="m",
        verdict=verdict,  # type: ignore[arg-type]
        confidence=kw.pop("confidence", "high"),
        before=before,
        after=after,
        **kw,
    )


def test_derive_labels_は既知の修正を含む候補を誤読とする() -> None:
    candidates = [
        _candidate("a", 60, _RAW[60]),
        _candidate("b", 70, "全く関係のない行"),
    ]
    labels = derive_labels(candidates, _KNOWN, {}, _RAW)
    assert labels == {"a": "misread", "b": "correct"}


def test_derive_labels_は手動ラベルで上書きする() -> None:
    candidates = [_candidate("b", 70, "全く関係のない行")]
    assert derive_labels(candidates, _KNOWN, {"b": "misread"}, _RAW) == {"b": "misread"}


def test_accuracy_report_は混同行列と修正の正しさを数える() -> None:
    candidates = [
        _candidate("tp", 60, _RAW[60]),
        _candidate("wrong_fix", 70, _RAW[70]),
        _candidate("fp", 60, "別の行。"),
        _candidate("tn", 60, "さらに別の行"),
        _candidate("fn", 60, "さらにさらに別の行"),
    ]
    labels = {
        "tp": "misread",
        "wrong_fix": "misread",
        "fp": "correct",
        "tn": "correct",
        "fn": "misread",
    }
    judgments = [
        _judgment("tp", "misread", "無制限に", "無限に"),
        _judgment("wrong_fix", "misread", "服部", "服辺"),
        _judgment("fp", "misread", "別の行", "別の列"),
        _judgment("tn", "correct"),
        _judgment("fn", "uncertain"),
    ]
    report = accuracy_report(candidates, judgments, labels, _RAW, _KNOWN)
    assert report.true_positive == 2  # 誤読を誤読と判定（修正の正誤は問わない）
    assert report.fix_correct == 1  # そのうち、既知の修正と同じ結果になったもの
    assert report.false_positive == 1
    assert report.true_negative == 1
    assert report.false_negative == 1  # uncertain は見逃し側に数える
    assert report.misread_count == 3
    assert report.correct_count == 2


def test_apply_to_pages_は修正を適用する() -> None:
    result = apply_to_pages(_RAW, (Correction(page=70, before="服部", after="隈部"),))
    assert result[70] == "著者は隈部正博です。\n"
    assert result[60] == _RAW[60]


def test_write_applied_pages_は後処理済みのページを書き出す(tmp_path: Path) -> None:
    write_applied_pages({3: "# 見出し\n本文\n"}, tmp_path)
    assert (tmp_path / "page_0003.md").read_text(encoding="utf-8").startswith("# 見出し")


def test_measure_scores_は修正の適用で指標が改善することを示す(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 3, "category": "地の文中心"}]}), encoding="utf-8"
    )
    (truth / "page_0003.md").write_text("著者は隈部正博です。\n", encoding="utf-8")
    raw_pages = {3: "著者は服部正博です。\n"}

    def no_error_checker(formulas: Any) -> list[str | None]:
        return [None for _ in formulas]

    before = measure_scores(truth, raw_pages, [], tmp_path / "a", katex_checker=no_error_checker)
    after = measure_scores(
        truth,
        raw_pages,
        [Correction(page=3, before="服部", after="隈部")],
        tmp_path / "b",
        katex_checker=no_error_checker,
    )
    assert before["prose_cer"] > 0
    assert after["prose_cer"] == 0


def test_accuracy_report_は数式の候補では候補の数式に含まれる既知の修正だけを期待値にする() -> None:
    raw = {70: "前の式 $(\\sqrt[3]{a})^4 > 1$ だから\n\n$$x = (\\sqrt[3]{a})^4 > y$$\n"}
    known = (
        Correction(page=70, before="$(\\sqrt[3]{a})^4 > 1$", after="$(\\sqrt[4]{a})^4 > 1$"),
        Correction(page=70, before="x = (\\sqrt[3]{a})^4 > y", after="x = (\\sqrt[4]{a})^4 > y"),
    )
    candidate = JudgeCandidate(
        id="eq",
        page=70,
        source="equation_check",
        kind="violated",
        text="x = (\\sqrt[3]{a})^4 > y",
        hint="",
        excerpt=raw[70],
        excerpt_scope="lines",
        line_index=2,
    )
    judgment = _judgment("eq", "misread", "x = (\\sqrt[3]{a})^4", "x = (\\sqrt[4]{a})^4")
    report = accuracy_report([candidate], [judgment], {"eq": "misread"}, raw, known)
    assert report.fix_correct == 1


def test_accuracy_report_は曖昧な置換前を抜粋で一意にして評価する() -> None:
    raw = {5: "無制限に繰り返す。\n\nそして無制限に続く。\n"}
    known = (Correction(page=5, before="無制限に繰り返す", after="無限に繰り返す"),)
    candidate = _candidate("c", 5, "無制限に繰り返す。")
    judgment = _judgment("c", "misread", "無制限", "無限")
    report = accuracy_report([candidate], [judgment], {"c": "misread"}, raw, known)
    assert report.fix_correct == 1
