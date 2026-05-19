"""仕様: MathEngineProtocol および NoOpMathEngine の動作検証。"""

from pathlib import Path

import pytest

from ouj_notebook_converter.plugins.math.base import MathEngineError, MathEngineProtocol
from ouj_notebook_converter.plugins.math.noop import NoOpMathEngine


class TestMathEngineProtocol:
    def test_NoOpMathEngineはMathEngineProtocolを満たす(self) -> None:
        assert isinstance(NoOpMathEngine(), MathEngineProtocol)

    def test_recognizeを持たないクラスはMathEngineProtocolを満たさない(self) -> None:
        class NoRecognize:
            pass

        assert not isinstance(NoRecognize(), MathEngineProtocol)


class TestNoOpMathEngine:
    def test_NoOpMathEngine_recognizeは空文字を返す(self) -> None:
        engine = NoOpMathEngine()
        result = engine.recognize(Path("dummy.png"))
        assert result == ""

    def test_MathEngineErrorはRuntimeErrorを継承する(self) -> None:
        with pytest.raises(RuntimeError):
            raise MathEngineError("テストエラー")
