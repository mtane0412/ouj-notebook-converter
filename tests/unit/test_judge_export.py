"""仕様: judge.policy / judge.export のユニットテスト。

判定結果を「自動適用」「人の確認待ち」「棄却」に振り分ける基準と、
修正ファイル（#28 の形式、version 1）への書き出しをテストする。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ouj_notebook_converter.corrections import apply_corrections, load_corrections
from ouj_notebook_converter.judge.export import build_exports, write_exports
from ouj_notebook_converter.judge.models import JudgeCandidate, Judgment
from ouj_notebook_converter.judge.policy import AutoPolicy, check_fix, combine, decide

_RAW = {
    60: "数の並びが無制限に繰り返される。\n別の行。\n",
    70: "(\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4",
}


def _candidate(**overrides: Any) -> JudgeCandidate:
    base: dict[str, Any] = {
        "id": "p0060-cross_ocr_diff-1",
        "page": 60,
        "source": "cross_ocr_diff",
        "kind": "gemini_only",
        "text": "制",
        "hint": "",
        "excerpt": _RAW[60],
        "excerpt_scope": "lines",
        "line_index": 0,
    }
    return JudgeCandidate(**{**base, **overrides})


def _judgment(**overrides: Any) -> Judgment:
    base: dict[str, Any] = {
        "candidate_id": "p0060-cross_ocr_diff-1",
        "model": "gemini-3.8-flash",
        "verdict": "misread",
        "confidence": "high",
        "before": "無制限に",
        "after": "無限に",
        "reason": "原本では「無限に」",
    }
    return Judgment(**{**base, **overrides})


class TestCheckFix:
    def test_一意に見つかれば_ok(self) -> None:
        assert check_fix(_judgment(), _RAW[60]) == "ok"

    def test_見つからなければ_not_found(self) -> None:
        assert check_fix(_judgment(before="存在しない"), _RAW[60]) == "not_found"

    def test_複数あれば_ambiguous(self) -> None:
        assert check_fix(_judgment(before="の"), _RAW[60]) == "ambiguous"

    def test_空または変化なしは不正(self) -> None:
        assert check_fix(_judgment(before="", after=""), _RAW[60]) == "empty"
        assert check_fix(_judgment(before="無制限に", after="無制限に"), _RAW[60]) == "unchanged"


class TestDecide:
    def test_確信度が高く小さな修正は自動適用する(self) -> None:
        decision, _ = decide(_candidate(), _judgment(), _RAW[60], AutoPolicy())
        assert decision == "auto"

    def test_確信度が高くなければ人の確認に回す(self) -> None:
        decision, reason = decide(
            _candidate(), _judgment(confidence="medium"), _RAW[60], AutoPolicy()
        )
        assert decision == "review"
        assert "確信度" in reason

    def test_正しいと判定した候補は棄却する(self) -> None:
        judgment = _judgment(verdict="correct", before="", after="")
        assert decide(_candidate(), judgment, _RAW[60], AutoPolicy())[0] == "reject"

    def test_正しいと判定しても確信度が低ければ人の確認に回す(self) -> None:
        judgment = _judgment(verdict="correct", confidence="low", before="", after="")
        assert decide(_candidate(), judgment, _RAW[60], AutoPolicy())[0] == "review"

    def test_判断不能と判定失敗は人の確認に回す(self) -> None:
        uncertain = _judgment(verdict="uncertain", before="", after="")
        failed = _judgment(verdict="uncertain", error="API 失敗")
        assert decide(_candidate(), uncertain, _RAW[60], AutoPolicy())[0] == "review"
        assert decide(_candidate(), failed, _RAW[60], AutoPolicy())[0] == "review"

    def test_置換前が一意でなければ人の確認に回す(self) -> None:
        judgment = _judgment(before="の", after="ノ")
        decision, reason = decide(_candidate(), judgment, _RAW[60], AutoPolicy())
        assert decision == "review"
        assert "置換前" in reason

    def test_大きな書き換えは自動適用しない(self) -> None:
        judgment = _judgment(
            before="数の並びが無制限に繰り返される", after="全く別の文章に書き換えた"
        )
        policy = AutoPolicy(max_edit_chars=4)
        assert decide(_candidate(), judgment, _RAW[60], policy)[0] == "review"

    def test_描画エラーの候補は既定では自動適用しない(self) -> None:
        candidate = _candidate(source="katex_error", kind="render_error")
        decision, reason = decide(candidate, _judgment(), _RAW[60], AutoPolicy())
        assert decision == "review"
        assert "katex_error" in reason

    def test_自動適用する検出器を限定できる(self) -> None:
        policy = AutoPolicy(auto_sources=frozenset({"equation_check"}))
        assert decide(_candidate(), _judgment(), _RAW[60], policy)[0] == "review"


class TestBuildExports:
    def test_自動適用と確認待ちを別の修正に振り分ける(self) -> None:
        candidates = [
            _candidate(),
            _candidate(id="p0070-equation_check-1", page=70, source="equation_check", text="x"),
        ]
        judgments = [
            _judgment(),
            _judgment(
                candidate_id="p0070-equation_check-1",
                confidence="medium",
                before="(\\sqrt[3]{a})^4",
                after="(\\sqrt[4]{a})^4",
            ),
        ]
        exports = build_exports(candidates, [judgments], _RAW, AutoPolicy(), date="2026-10-10")
        assert [c.page for c in exports.auto] == [60]
        assert exports.auto[0].reviewer == "llm-judge:gemini-3.8-flash"
        assert exports.auto[0].date == "2026-10-10"
        assert "p0060-cross_ocr_diff-1" in exports.auto[0].reason
        assert [c.page for c in exports.pending] == [70]
        assert exports.pending[0].reviewer == ""
        decisions = {d["candidate_id"]: d["decision"] for d in exports.decisions}
        assert decisions == {"p0060-cross_ocr_diff-1": "auto", "p0070-equation_check-1": "review"}

    def test_同じ結果になる重複した修正は片方だけ書き出す(self) -> None:
        candidates = [_candidate(), _candidate(id="p0060-rare_word-1", source="rare_word")]
        judgments = [_judgment(), _judgment(candidate_id="p0060-rare_word-1")]
        exports = build_exports(candidates, [judgments], _RAW, AutoPolicy(), date="2026-10-10")
        assert len(exports.auto) == 1
        decisions = [d for d in exports.decisions if d["candidate_id"] == "p0060-rare_word-1"]
        assert decisions[0]["decision"] == "auto"
        assert "適用済み" in decisions[0]["reason"]

    def test_同じ箇所を別の内容に直す修正は確認待ちにする(self) -> None:
        candidates = [_candidate(), _candidate(id="p0060-rare_word-1", source="rare_word")]
        judgments = [
            _judgment(),
            _judgment(
                candidate_id="p0060-rare_word-1", before="無制限に繰り返", after="無制約に繰り返"
            ),
        ]
        exports = build_exports(candidates, [judgments], _RAW, AutoPolicy(), date="2026-10-10")
        assert len(exports.auto) == 1
        decisions = [d for d in exports.decisions if d["candidate_id"] == "p0060-rare_word-1"]
        assert decisions[0]["decision"] == "review"
        assert "重複" in decisions[0]["reason"] or "重なる" in decisions[0]["reason"]

    def test_ページ内で曖昧な置換前は抜粋の位置を手がかりに一意にする(self) -> None:
        raw = {5: "無制限に繰り返す。\n\nそして無制限に続く。\n"}
        candidate = _candidate(
            id="p0005-rare_word-1",
            page=5,
            excerpt="無制限に繰り返す。",
            excerpt_scope="lines",
        )
        judgment = _judgment(candidate_id="p0005-rare_word-1", before="無制限", after="無限")
        exports = build_exports([candidate], [[judgment]], raw, AutoPolicy(), date="2026-10-10")
        assert len(exports.auto) == 1
        correction = exports.auto[0]
        assert raw[5].count(correction.before) == 1
        assert (
            apply_corrections(raw[5], (correction,)) == "無限に繰り返す。\n\nそして無制限に続く。\n"
        )

    def test_判定の無い候補は例外(self) -> None:
        with pytest.raises(ValueError, match="判定結果が無い"):
            build_exports([_candidate()], [[]], _RAW, AutoPolicy(), date="2026-10-10")

    def test_書き出した修正ファイルは_corrections_の形式で読み込め適用できる(
        self, tmp_path: Path
    ) -> None:
        exports = build_exports(
            [_candidate()], [[_judgment()]], _RAW, AutoPolicy(), date="2026-10-10"
        )
        write_exports(exports, tmp_path)
        loaded = load_corrections(tmp_path / "corrections.json")
        assert (
            apply_corrections(_RAW[60], loaded[60]) == "数の並びが無限に繰り返される。\n別の行。\n"
        )
        assert json.loads((tmp_path / "corrections.json").read_text())["version"] == 1
        # 確認待ちが無くても形式の正しいファイルを書く
        assert load_corrections(tmp_path / "corrections_pending.json") == {}
        assert (tmp_path / "decisions.json").is_file()


class TestCombine:
    """複数回の判定（切り出し画像とページ全体など）を 1 つの判定にまとめる。"""

    def test_全員が同じ修正なら誤読として一致数を数える(self) -> None:
        crop = _judgment(model="crop")
        page = _judgment(model="page", before="無制限に繰り返", after="無限に繰り返")
        combined = combine([crop, page], _candidate(), _RAW[60])
        assert combined.judgment.verdict == "misread"
        assert combined.agree == 2

    def test_修正の結果が同じなら範囲が違っても一致とみなす(self) -> None:
        crop = _judgment(before="無制限に", after="無限に")
        page = _judgment(before="が無制限に繰り返", after="が無限に繰り返")
        assert combine([crop, page], _candidate(), _RAW[60]).agree == 2

    def test_誤読と正しいで割れたら判断不能にする(self) -> None:
        crop = _judgment()
        page = _judgment(verdict="correct", before="", after="")
        combined = combine([crop, page], _candidate(), _RAW[60])
        assert combined.judgment.verdict == "uncertain"
        assert "割れ" in combined.judgment.reason
        assert combined.agree == 0

    def test_誤読でも修正が違えば判断不能にする(self) -> None:
        crop = _judgment(after="無限に")
        page = _judgment(after="無数に")
        assert combine([crop, page], _candidate(), _RAW[60]).judgment.verdict == "uncertain"

    def test_判断不能は棄権として数えない(self) -> None:
        crop = _judgment(verdict="uncertain", before="", after="")
        page = _judgment(verdict="correct", before="", after="")
        combined = combine([crop, page], _candidate(), _RAW[60])
        assert combined.judgment.verdict == "correct"
        assert combined.agree == 1

    def test_全員が正しいなら確信度は最も低いものにする(self) -> None:
        crop = _judgment(verdict="correct", confidence="high", before="", after="")
        page = _judgment(verdict="correct", confidence="medium", before="", after="")
        assert combine([crop, page], _candidate(), _RAW[60]).judgment.confidence == "medium"

    def test_全員が判断不能なら判断不能(self) -> None:
        crop = _judgment(verdict="uncertain", before="", after="")
        assert combine([crop], _candidate(), _RAW[60]).judgment.verdict == "uncertain"


class TestMinAgree:
    def test_一致数が足りない誤読は自動適用しない(self) -> None:
        policy = AutoPolicy(min_agree=2)
        decision, reason = decide(_candidate(), _judgment(), _RAW[60], policy, agree=1)
        assert decision == "review"
        assert "一致" in reason

    def test_一致数が足りれば自動適用する(self) -> None:
        policy = AutoPolicy(min_agree=2)
        assert decide(_candidate(), _judgment(), _RAW[60], policy, agree=2)[0] == "auto"

    def test_複数回の判定の一致で自動適用に進む(self) -> None:
        candidates = [_candidate()]
        runs = [[_judgment(model="crop")], [_judgment(model="page")]]
        exports = build_exports(candidates, runs, _RAW, AutoPolicy(min_agree=2), date="2026-10-10")
        assert len(exports.auto) == 1

    def test_片方の判定だけでは確認待ちになる(self) -> None:
        candidates = [_candidate()]
        runs = [
            [_judgment(model="crop")],
            [_judgment(model="page", verdict="correct", before="", after="")],
        ]
        exports = build_exports(candidates, runs, _RAW, AutoPolicy(min_agree=2), date="2026-10-10")
        assert exports.auto == ()
        assert exports.decisions[0]["decision"] == "review"
