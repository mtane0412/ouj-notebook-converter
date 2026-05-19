"""仕様: 数式変換を行わない NoOp 実装。--math 未指定時の規定値およびテスト用。"""

from __future__ import annotations

from pathlib import Path


class NoOpMathEngine:
    """常に空文字列を返す数式エンジン。変換をスキップしたい場合に使用する。"""

    def recognize(self, image_path: Path) -> str:
        """何もせず空文字列を返す。"""
        return ""
