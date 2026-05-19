"""仕様: 数式画像 → LaTeX 変換エンジンの Protocol・例外・データ型を定義する。"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FormulaDetection:
    """Pix2Text 検出結果 1 件。

    box: (x1, y1, x2, y2) ピクセル絶対座標
    type: "isolated" (display 式) / "embedding" (inline 式)
    latex: 認識済み LaTeX ソース（先頭末尾の空白除去済み）
    score: 検出スコア (0–1)
    """

    box: tuple[int, int, int, int]
    type: str
    latex: str
    score: float


@runtime_checkable
class MathRecognizerProtocol(Protocol):
    """crop 済み数式画像 1 枚を LaTeX 文字列とスコアに変換するエンジンのインターフェース。

    日本語ラベルを除外してトリミングした画像の再認識に使用する。
    失敗時は MathEngineError を送出する（暗黙フォールバック禁止）。
    """

    def recognize_image(self, image_path: Path) -> tuple[str, float]:
        """PNG 画像 1 枚を LaTeX 文字列とスコアのタプルに変換する。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            (latex, score) のタプル。latex は先頭末尾の空白除去済み。

        Raises:
            MathEngineError: エンジンの呼び出しに失敗した場合。
        """
        ...


@runtime_checkable
class MathDetectorProtocol(Protocol):
    """ページ画像 1 枚から数式 bbox と LaTeX を一括取得するエンジンのインターフェース。

    失敗時は MathEngineError を送出する（暗黙フォールバック禁止）。
    """

    def detect_and_recognize(self, image_path: Path) -> list[FormulaDetection]:
        """PNG 画像 1 枚から数式を検出し、LaTeX を認識して返す。

        Args:
            image_path: 入力画像の絶対パス（PNG 形式）。

        Returns:
            検出された数式ごとの FormulaDetection リスト。
            数式なしの場合は空リスト。

        Raises:
            MathEngineError: エンジンの呼び出しに失敗した場合。
        """
        ...
