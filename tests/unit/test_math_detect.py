"""仕様: math_detect ステージ（Pix2Text による数式検出・IoU マッチ・MathOverlay 構築）の動作検証。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.stages.math_detect import (
    iou,
    match_paragraph_by_iou,
    math_detect,
)
from ouj_notebook_converter.pipeline.types import PageAnalysis
from ouj_notebook_converter.plugins.math.base import FormulaDetection


def _make_analysis_json(tmp_path: Path, paragraphs: list[dict[str, Any]]) -> Path:
    """テスト用の analysis.json を作成するヘルパ。"""
    data = {"paragraphs": paragraphs, "tables": [], "words": [], "figures": []}
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_page_analysis(tmp_path: Path, paragraphs: list[dict[str, Any]]) -> PageAnalysis:
    """テスト用の PageAnalysis を作成するヘルパ。"""
    json_path = _make_analysis_json(tmp_path, paragraphs)
    return PageAnalysis(
        page_index=0,
        yomitoku_json_path=json_path,
        markdown_raw_path=tmp_path / "raw.md",
    )


class FakeMathDetector:
    """テスト用の固定 FormulaDetection リストを返す検出エンジン。"""

    def __init__(self, detections: list[FormulaDetection] | None = None) -> None:
        self._detections = detections or []
        self.called_paths: list[Path] = []

    def detect_and_recognize(self, image_path: Path) -> list[FormulaDetection]:
        self.called_paths.append(image_path)
        return self._detections


class TestIou:
    def test_iouは完全一致で1を返す(self) -> None:
        assert iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_iouは交差なしで0を返す(self) -> None:
        assert iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_iouは半分重なる場合の値が正しい(self) -> None:
        # box_a: (0,0)-(100,100) area=10000
        # box_b: (50,0)-(150,100) area=10000
        # intersection: (50,0)-(100,100) area=5000
        # union: 15000
        result = iou((0, 0, 100, 100), (50, 0, 150, 100))
        assert result == pytest.approx(5000 / 15000, rel=1e-3)

    def test_iouは片方がゼロサイズでも0を返す(self) -> None:
        # ゼロサイズ box（面積 0）の場合は IoU は定義できないため 0 を返す
        assert iou((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


class TestMatchParagraphByIou:
    def test_match_paragraph_by_iouはIoU最大のparagraphを返す(self) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 100], "contents": "段落A", "role": None, "order": 0},
            {"box": [200, 0, 300, 100], "contents": "段落B", "role": None, "order": 1},
        ]
        # detection box は段落A と完全一致
        result = match_paragraph_by_iou((0, 0, 100, 100), paragraphs, threshold=0.3)
        assert result is not None
        assert result["contents"] == "段落A"

    def test_match_paragraph_by_iouは閾値未満ならNoneを返す(self) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 100], "contents": "段落C", "role": None, "order": 0},
        ]
        # detection box はほとんど重ならない
        result = match_paragraph_by_iou((500, 500, 600, 600), paragraphs, threshold=0.3)
        assert result is None

    def test_match_paragraph_by_iouはparagraphsが空ならNoneを返す(self) -> None:
        result = match_paragraph_by_iou((0, 0, 100, 100), [], threshold=0.3)
        assert result is None

    def test_match_paragraph_by_iouはcontentsがNoneのparagraphを無視する(self) -> None:
        paragraphs: list[dict[str, Any]] = [
            {"box": [0, 0, 100, 100], "contents": None, "role": None, "order": 0},
            {"box": [0, 0, 100, 100], "contents": "段落D", "role": None, "order": 1},
        ]
        result = match_paragraph_by_iou((0, 0, 100, 100), paragraphs, threshold=0.3)
        assert result is not None
        assert result["contents"] == "段落D"


class TestMathDetect:
    def test_math_detectは検出ゼロで空のMathOverlayを返す(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        analysis = _make_page_analysis(tmp_path, [])
        detector = FakeMathDetector(detections=[])
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector)

        assert result.items == {}
        assert result.roles == {}
        assert result.originals == {}

    def test_math_detectはdetector_detect_and_recognizeを1回呼ぶ(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        analysis = _make_page_analysis(tmp_path, [])
        detector = FakeMathDetector(detections=[])
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        math_detect(image, analysis, tmp_path, detector=detector)

        assert len(detector.called_paths) == 1

    def test_math_detectはマッチしたparagraphのcontentsをoriginalsに格納する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [10, 20, 100, 80], "contents": "E=mc^2", "role": None, "order": 0},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        detections = [
            FormulaDetection(box=(10, 20, 100, 80), type="isolated", latex=r"E=mc^2", score=0.9),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector)

        # MathOverlay に 1 件のエントリが登録されていること
        assert len(result.originals) == 1
        assert "E=mc^2" in result.originals.values()

    def test_math_detectはembeddingをinline_formulaに変換する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 50], "contents": "インライン数式", "role": None, "order": 0},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        detections = [
            FormulaDetection(box=(0, 0, 100, 50), type="embedding", latex=r"x^2", score=0.85),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector)

        assert "inline_formula" in result.roles.values()

    def test_math_detectはisolatedをdisplay_formulaに変換する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 200, 100], "contents": "ディスプレイ数式", "role": None, "order": 0},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        detections = [
            FormulaDetection(
                box=(0, 0, 200, 100), type="isolated", latex=r"\frac{1}{2}", score=0.92
            ),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector)

        assert "display_formula" in result.roles.values()

    def test_math_detectはマッチしない検出をスキップしwarningを出す(
        self, tmp_path: Path, mocker: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # paragraph は別の場所にある
        paragraphs = [
            {"box": [0, 0, 50, 50], "contents": "遠い段落", "role": None, "order": 0},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        # detection は画像の全く別の場所
        detections = [
            FormulaDetection(box=(500, 500, 600, 600), type="isolated", latex=r"\pi", score=0.7),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING):
            result = math_detect(image, analysis, tmp_path, detector=detector)

        assert result.items == {}
        assert any("スキップ" in r.message for r in caplog.records)

    def test_math_detectはクロップPNGを連番でmath_ディレクトリに保存する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 50], "contents": "数式A", "role": None, "order": 0},
            {"box": [0, 60, 100, 110], "contents": "数式B", "role": None, "order": 1},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        detections = [
            FormulaDetection(box=(0, 0, 100, 50), type="embedding", latex="a", score=0.9),
            FormulaDetection(box=(0, 60, 100, 110), type="isolated", latex="b", score=0.8),
        ]
        detector = FakeMathDetector(detections=detections)
        # page.png 保存と crop_math_image 内の保存を両方モック
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector)

        # 2 件の数式が MathOverlay に登録されていること
        assert len(result.items) == 2
        assert len(result.roles) == 2
        assert len(result.originals) == 2
