"""仕様: KaTeX で数式を描画できるかを検査する。

KaTeX は JavaScript 製のため、Node.js の検査スクリプト（scripts/katex_check/check.mjs）を
subprocess で 1 回だけ起動し、全数式を JSON でまとめて渡す。

事前準備:
  cd scripts/katex_check && npm install

注意事項:
  - Node.js や katex が無い場合は検査を省略せず KatexCheckError を送出する（Fail-Fast）
  - 検査スクリプトの位置はリポジトリ構成（src/ouj_notebook_converter/evaluation/katex.py から
    4 階層上がリポジトリルート）を前提とする
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ouj_notebook_converter.evaluation.markdown_parts import Formula

KATEX_CHECK_DIR = Path(__file__).resolve().parents[3] / "scripts" / "katex_check"
_KATEX_CHECK_SCRIPT = KATEX_CHECK_DIR / "check.mjs"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class KatexCheckError(RuntimeError):
    """KaTeX 検査スクリプトを実行できなかった場合の例外。"""


def check_katex(
    formulas: Sequence[Formula], *, runner: Runner = subprocess.run
) -> list[str | None]:
    """数式ごとに KaTeX で描画できるかを検査する。

    Args:
        formulas: 検査する数式。
        runner: 外部プロセスの実行関数（テストで差し替える）。

    Returns:
        formulas と同じ順序・件数のリスト。描画できれば None、できなければ KaTeX のエラーメッセージ。

    Raises:
        KatexCheckError: Node.js が無い、検査スクリプトが異常終了した、結果の形式が不正な場合。
    """
    if not formulas:
        return []
    payload = json.dumps([{"tex": f.tex, "display": f.display} for f in formulas])
    try:
        completed = runner(
            ["node", str(_KATEX_CHECK_SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError as e:
        raise KatexCheckError("Node.js（node コマンド）が見つかりません") from e
    if completed.returncode != 0:
        raise KatexCheckError(
            f"KaTeX 検査スクリプトが異常終了しました（{KATEX_CHECK_DIR} で npm install 済みか確認してください）: "
            f"{completed.stderr.strip()}"
        )
    return _parse_results(completed.stdout, expected_count=len(formulas))


def _parse_results(stdout: str, *, expected_count: int) -> list[str | None]:
    """検査スクリプトの出力（エラーメッセージまたは null の JSON 配列）を検証して返す。"""
    try:
        results: Any = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise KatexCheckError(
            f"KaTeX 検査スクリプトの出力が JSON ではありません: {stdout!r}"
        ) from e
    if not isinstance(results, list) or not all(r is None or isinstance(r, str) for r in results):
        raise KatexCheckError(f"KaTeX 検査スクリプトの出力形式が不正です: {stdout!r}")
    if len(results) != expected_count:
        raise KatexCheckError(
            f"KaTeX 検査結果の件数（{len(results)}）が数式の件数（{expected_count}）と一致しません"
        )
    return results
