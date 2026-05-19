"""仕様: MathEngineProtocol / NoOpMathEngine および MathDetectorProtocol / FormulaDetection の動作検証。"""

from pathlib import Path

import pytest

from ouj_notebook_converter.plugins.math.base import (
    FormulaDetection,
    MathDetectorProtocol,
    MathEngineError,
    MathEngineProtocol,
)
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


class TestFormulaDetection:
    def test_FormulaDetectionが各フィールドを保持する(self) -> None:
        det = FormulaDetection(
            box=(10, 20, 100, 80),
            type="isolated",
            latex=r"\frac{1}{2}",
            score=0.95,
        )
        assert det.box == (10, 20, 100, 80)
        assert det.type == "isolated"
        assert det.latex == r"\frac{1}{2}"
        assert det.score == 0.95

    def test_FormulaDetectionはimmutable(self) -> None:
        det = FormulaDetection(box=(0, 0, 10, 10), type="embedding", latex="x", score=0.8)
        with pytest.raises(AttributeError):
            det.latex = "y"  # type: ignore[misc]


class TestMathDetectorProtocol:
    def test_detect_and_recognizeを持つクラスはMathDetectorProtocolを満たす(self) -> None:
        class FakeDetector:
            def detect_and_recognize(self, image_path: Path) -> list[FormulaDetection]:
                return []

        assert isinstance(FakeDetector(), MathDetectorProtocol)

    def test_detect_and_recognizeを持たないクラスはMathDetectorProtocolを満たさない(self) -> None:
        class NoDetect:
            pass

        assert not isinstance(NoDetect(), MathDetectorProtocol)
