"""仕様: evaluation.katex モジュール（KaTeX による数式の描画可否検査）のユニットテスト。

Node.js の検査スクリプトを subprocess で呼び出す check_katex をテストする。
外部プロセスはフェイクに差し替え、実際の KaTeX を使う検査は katex がインストール済みの
場合のみ実行する。
"""

import json
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from ouj_notebook_converter.evaluation.katex import (
    KATEX_CHECK_DIR,
    KatexCheckError,
    check_katex,
)
from ouj_notebook_converter.evaluation.markdown_parts import Formula


class _FakeRunner:
    """subprocess.run の代わりに、受け取った入力を記録して決まった結果を返すフェイク。"""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.received_input: str | None = None

    def __call__(self, args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.received_input = kwargs["input"]
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


class TestCheckKatex:
    """check_katex の入出力。"""

    def test_returns_error_message_per_formula(self) -> None:
        """数式ごとに、描画できれば None、できなければエラーメッセージを返す。"""
        runner = _FakeRunner(stdout=json.dumps([None, "Undefined control sequence: \\cline"]))
        formulas = [Formula(tex="x", display=False), Formula(tex="\\cline{1-2}", display=True)]

        assert check_katex(formulas, runner=runner) == [
            None,
            "Undefined control sequence: \\cline",
        ]
        assert json.loads(runner.received_input or "") == [
            {"tex": "x", "display": False},
            {"tex": "\\cline{1-2}", "display": True},
        ]

    def test_empty_input_does_not_spawn_process(self) -> None:
        """数式が無ければ外部プロセスを起動せず空リストを返す。"""
        runner = _FakeRunner(stdout="")

        assert check_katex([], runner=runner) == []
        assert runner.received_input is None

    def test_process_failure_raises(self) -> None:
        """検査スクリプトが異常終了すれば標準エラー出力を含めて KatexCheckError を送出する。"""
        runner = _FakeRunner(stdout="", returncode=1, stderr="Cannot find package 'katex'")

        with pytest.raises(KatexCheckError, match="Cannot find package 'katex'"):
            check_katex([Formula(tex="x", display=False)], runner=runner)

    def test_result_count_mismatch_raises(self) -> None:
        """結果の件数が数式の件数と合わなければ KatexCheckError を送出する。"""
        runner = _FakeRunner(stdout=json.dumps([None]))
        formulas = [Formula(tex="x", display=False), Formula(tex="y", display=False)]

        with pytest.raises(KatexCheckError, match="件数"):
            check_katex(formulas, runner=runner)


@pytest.mark.skipif(
    shutil.which("node") is None or not (KATEX_CHECK_DIR / "node_modules" / "katex").exists(),
    reason="Node.js または katex が未インストール（scripts/katex_check で npm install が必要）",
)
def test_real_katex_detects_unsupported_command() -> None:
    """実際の KaTeX で、描画できる数式と描画できない数式を判別できる。"""
    results = check_katex(
        [
            Formula(tex="\\sqrt[4]{a}", display=False),
            Formula(tex="\\begin{array}{r}12\\\\ \\cline{1-1}\\end{array}", display=True),
        ]
    )

    assert results[0] is None
    assert results[1] is not None
