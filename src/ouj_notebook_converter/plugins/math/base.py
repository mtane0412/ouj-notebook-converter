"""仕様: 数式画像 → LaTeX 変換エンジンの Protocol と例外を定義する。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class MathEngineError(RuntimeError):
    """数式エンジンの呼び出し失敗を示す例外。暗黙フォールバックの代わりに送出する。"""


@runtime_checkable
class MathEngineProtocol(Protocol):
    """数式画像 1 枚を LaTeX 文字列に変換するエンジンのインターフェース。

    失敗時は MathEngineError を送出する（暗黙フォールバック禁止）。
    空文字列は「変換しなかった (NoOp)」を意味し、math_extract 側でスキップする。
    """

    def recognize(self, image_path: Path) -> str:
        """PNG 画像 1 枚を LaTeX 文字列に変換する。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            LaTeX ソース文字列（先頭末尾の空白は除去済み）。
            空文字列は「変換しなかった」を意味する。

        Raises:
            MathEngineError: エンジンの呼び出しに失敗した場合。
        """
        ...
