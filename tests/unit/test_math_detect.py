"""仕様: math_detect ステージ（Pix2Text による数式検出・IoA マッチ・MathOverlay 構築）の動作検証。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.stages.math_detect import (
    ioa,
    iou,
    is_japanese_text,
    match_paragraph_by_ioa,
    math_detect,
    trim_bbox_by_japanese_words,
    word_points_to_box,
)
from ouj_notebook_converter.pipeline.types import PageAnalysis
from ouj_notebook_converter.plugins.math.base import FormulaDetection, MathEngineError


def _make_word_dict(x1: int, y1: int, x2: int, y2: int, content: str) -> dict[str, Any]:
    """テスト用 word dict（軸並行矩形の 4 点 points 形式）を作成するヘルパ。"""
    return {
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "content": content,
    }


def _make_analysis_json(
    tmp_path: Path,
    paragraphs: list[dict[str, Any]],
    words: list[dict[str, Any]] | None = None,
) -> Path:
    """テスト用の analysis.json を作成するヘルパ。"""
    data = {"paragraphs": paragraphs, "tables": [], "words": words or [], "figures": []}
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_page_analysis(
    tmp_path: Path,
    paragraphs: list[dict[str, Any]],
    words: list[dict[str, Any]] | None = None,
) -> PageAnalysis:
    """テスト用の PageAnalysis を作成するヘルパ。"""
    json_path = _make_analysis_json(tmp_path, paragraphs, words)
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


class FakeMathRecognizer:
    """テスト用の固定 LaTeX を返す再認識エンジン。"""

    def __init__(self, latex: str = r"\trimmed", score: float = 0.99) -> None:
        self._latex = latex
        self._score = score
        self.called_paths: list[Path] = []

    def recognize_image(self, image_path: Path) -> tuple[str, float]:
        self.called_paths.append(image_path)
        return (self._latex, self._score)


class FakeFailingRecognizer:
    """recognize_image が常に MathEngineError を送出するエンジン。"""

    def recognize_image(self, image_path: Path) -> tuple[str, float]:
        raise MathEngineError("再認識失敗")


class TestIsJapaneseText:
    def test_ひらがなを含めばTrueを返す(self) -> None:
        assert is_japanese_text("最大化") is True

    def test_カタカナを含めばTrueを返す(self) -> None:
        assert is_japanese_text("コスト") is True

    def test_漢字を含めばTrueを返す(self) -> None:
        assert is_japanese_text("制約条件") is True

    def test_ひらがなを含む混合テキストはTrueを返す(self) -> None:
        assert is_japanese_text("z = 1 のとき") is True

    def test_ASCII英数字のみはFalseを返す(self) -> None:
        assert is_japanese_text("x = 1 + 2") is False

    def test_数式記号のみはFalseを返す(self) -> None:
        # ∑ ≤ ∈ ∀ は CJK 範囲外なので False
        assert is_japanese_text("∑≤∈∀") is False

    def test_空文字列はFalseを返す(self) -> None:
        assert is_japanese_text("") is False

    def test_LaTeXコマンドのみはFalseを返す(self) -> None:
        assert is_japanese_text(r"\frac{1}{2}") is False


class TestWordPointsToBox:
    def test_正方形ポリゴンをbboxに変換する(self) -> None:
        # 左上(10,20)、右上(110,20)、右下(110,80)、左下(10,80) の正方形
        points = [[10, 20], [110, 20], [110, 80], [10, 80]]
        assert word_points_to_box(points) == (10, 20, 110, 80)

    def test_回転したポリゴンからaxisaligned_bboxを返す(self) -> None:
        # 45 度回転したひし形 (中心(50,50)、外接矩形は(30,30)-(70,70))
        points = [[50, 30], [70, 50], [50, 70], [30, 50]]
        assert word_points_to_box(points) == (30, 30, 70, 70)

    def test_単一点ポリゴンはゼロサイズbboxを返す(self) -> None:
        points = [[5, 5], [5, 5], [5, 5], [5, 5]]
        assert word_points_to_box(points) == (5, 5, 5, 5)


class TestTrimBboxByJapaneseWords:
    """trim_bbox_by_japanese_words の純粋関数テスト。"""

    @staticmethod
    def _make_word(x1: int, y1: int, x2: int, y2: int, content: str) -> dict:
        """テスト用 word dict（points は軸並行矩形の 4 点）を作成する。"""
        return {
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "content": content,
        }

    def test_日本語wordがない場合は元のboxをそのまま返す(self) -> None:
        formula_box = (0, 0, 200, 100)
        words = [self._make_word(10, 10, 190, 40, "z = x_1 + 2")]
        result = trim_bbox_by_japanese_words(formula_box, words)
        assert result == formula_box

    def test_最大化ラベル行を除外したbboxを返す(self) -> None:
        # formula_box は日本語ラベル行（y=0-30）と数式行（y=40-90）を含む
        formula_box = (0, 0, 200, 100)
        words = [
            self._make_word(0, 0, 80, 30, "最大化"),      # 日本語ラベル
            self._make_word(90, 40, 200, 90, "z = x_1"),  # 数式部分
        ]
        result = trim_bbox_by_japanese_words(formula_box, words)
        # 数式部分の bbox (90, 40, 200, 90) が返ることを期待
        assert result is not None
        assert result[0] == 90   # x1
        assert result[1] == 40   # y1
        assert result[2] == 200  # x2
        assert result[3] == 90   # y2

    def test_制約条件ラベルを除外したbboxを返す(self) -> None:
        formula_box = (0, 0, 300, 200)
        words = [
            self._make_word(0, 0, 70, 30, "制約条件"),
            self._make_word(0, 50, 60, 80, "x_1"),
            self._make_word(70, 50, 140, 80, "+ 3x_2"),
            self._make_word(150, 50, 200, 80, "≤ 24"),
        ]
        result = trim_bbox_by_japanese_words(formula_box, words)
        assert result is not None
        # 日本語ラベルが除外され、数式部分のみの bbox になること
        assert result[1] > 30  # 日本語ラベル行（y=0-30）は除外される

    def test_数式記号のみのwordは除外対象にならない(self) -> None:
        # ∑ ∈ ≤ は CJK 範囲外なので non_japanese 扱いになる
        formula_box = (0, 0, 200, 100)
        words = [
            self._make_word(0, 0, 50, 50, "∑"),
            self._make_word(60, 0, 150, 50, "∈"),
        ]
        result = trim_bbox_by_japanese_words(formula_box, words)
        # 日本語 word がないので元の formula_box と同じ bbox（または数式 words の範囲）を返す
        assert result is not None

    def test_全てが日本語wordの場合はNoneを返す(self) -> None:
        formula_box = (0, 0, 200, 100)
        words = [
            self._make_word(0, 0, 80, 40, "最大化"),
            self._make_word(0, 50, 100, 90, "制約条件"),
        ]
        result = trim_bbox_by_japanese_words(formula_box, words)
        assert result is None

    def test_overlap_ioa閾値未満のwordは日本語でも無視する(self) -> None:
        # formula_box から遠い位置にある日本語 word は除外対象にならない
        formula_box = (0, 0, 100, 100)
        words = [
            self._make_word(0, 0, 80, 50, "z = 1"),       # 数式 word（formula 内）
            self._make_word(500, 500, 600, 550, "最大化"),  # formula から遠い日本語 word
        ]
        result = trim_bbox_by_japanese_words(formula_box, words)
        assert result is not None

    def test_wordsが空ならformula_boxをそのまま返す(self) -> None:
        formula_box = (10, 20, 110, 80)
        result = trim_bbox_by_japanese_words(formula_box, [])
        assert result == formula_box

    def test_トリミング結果が幅0になる場合はNoneを返す(self) -> None:
        # non_japanese_words の x 幅が 0 になるケース
        formula_box = (50, 0, 50, 100)  # 幅 0 の formula_box
        words = [self._make_word(50, 0, 50, 100, "z = 1")]  # 幅 0
        result = trim_bbox_by_japanese_words(formula_box, words)
        assert result is None


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


class TestIoa:
    def test_ioaは数式が段落に完全包含されると1を返す(self) -> None:
        # 数式 bbox が paragraph bbox に完全に含まれる
        assert ioa((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_ioaは交差なしで0を返す(self) -> None:
        assert ioa((200, 200, 300, 300), (0, 0, 100, 100)) == pytest.approx(0.0)

    def test_ioaは半分はみ出す場合0_5を返す(self) -> None:
        # 数式 (0,0)-(100,100) area=10000 の半分が段落 (50,0)-(200,100) に含まれる
        # intersection: (50,0)-(100,100) area=5000
        # IoA = 5000 / 10000 = 0.5
        assert ioa((0, 0, 100, 100), (50, 0, 200, 100)) == pytest.approx(0.5)

    def test_ioaは数式がゼロサイズなら0を返す(self) -> None:
        # 数式面積がゼロの場合は定義できないため 0 を返す
        assert ioa((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


class TestMatchParagraphByIoa:
    def test_match_paragraph_by_ioaはIoA最大のparagraphを返す(self) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 100], "contents": "段落A", "role": None, "order": 0},
            {"box": [200, 0, 300, 100], "contents": "段落B", "role": None, "order": 1},
        ]
        # detection box は段落A に完全包含
        result = match_paragraph_by_ioa((0, 0, 100, 100), paragraphs, threshold=0.5)
        assert result is not None
        assert result["contents"] == "段落A"

    def test_インライン数式は幅広い段落に完全包含されればマッチする(self) -> None:
        # 幅広い paragraph の中に小さなインライン数式が含まれるケース
        # IoU では IoA ≪ 0.03 でマッチしないが、IoA = 1.0 でマッチする
        paragraphs = [
            {"box": [0, 0, 900, 100], "contents": "z は定数である", "role": None, "order": 0},
        ]
        result = match_paragraph_by_ioa((400, 10, 430, 90), paragraphs, threshold=0.5)
        assert result is not None
        assert result["contents"] == "z は定数である"

    def test_match_paragraph_by_ioaは閾値未満ならNoneを返す(self) -> None:
        paragraphs = [
            {"box": [0, 0, 100, 100], "contents": "段落C", "role": None, "order": 0},
        ]
        # detection box はほとんど重ならない
        result = match_paragraph_by_ioa((500, 500, 600, 600), paragraphs, threshold=0.5)
        assert result is None

    def test_match_paragraph_by_ioaはparagraphsが空ならNoneを返す(self) -> None:
        result = match_paragraph_by_ioa((0, 0, 100, 100), [], threshold=0.5)
        assert result is None

    def test_match_paragraph_by_ioaはcontentsがNoneのparagraphを無視する(self) -> None:
        paragraphs: list[dict[str, Any]] = [
            {"box": [0, 0, 100, 100], "contents": None, "role": None, "order": 0},
            {"box": [0, 0, 100, 100], "contents": "段落D", "role": None, "order": 1},
        ]
        result = match_paragraph_by_ioa((0, 0, 100, 100), paragraphs, threshold=0.5)
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
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

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
        math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

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
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # MathOverlay に 1 件のエントリが登録されていること
        assert len(result.originals) == 1
        assert "E=mc^2" in result.originals.values()

    def test_math_detectはembeddingをスキップしてMathOverlayに登録しない(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # embedding（インライン数式）は段落全体置換ができないためスキップする
        paragraphs = [
            {"box": [0, 0, 100, 50], "contents": "インライン数式が含まれる段落", "role": None, "order": 0},
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        detections = [
            FormulaDetection(box=(0, 0, 100, 50), type="embedding", latex=r"x^2", score=0.85),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # embedding はスキップされるため MathOverlay は空になる
        assert result.items == {}

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
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

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
            result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        assert result.items == {}
        assert any("スキップ" in r.message for r in caplog.records)

    def test_math_detectはisolatedのみMathOverlayに登録する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # isolated 1 件 + embedding 1 件 → isolated のみ登録、embedding はスキップ
        paragraphs = [
            {"box": [0, 0, 100, 50], "contents": "インライン数式の段落", "role": None, "order": 0},
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
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # isolated 1 件のみ登録されていること（embedding はスキップ）
        assert len(result.items) == 1
        assert len(result.roles) == 1
        assert len(result.originals) == 1
        assert "display_formula" in result.roles.values()


class TestMathDetectWithJapaneseTrimming:
    """日本語ラベル混入時の bbox トリミング統合テスト。"""

    def test_日本語ラベル付き数式はトリミング後にrecognizerが呼ばれる(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # formula_box 内に「最大化」（日本語ラベル、y=0-30）と
        # 「z = x_1 + 2」（数式、y=50-90）の 2 つの word がある
        paragraphs = [
            {"box": [0, 0, 200, 100], "contents": "最大化 z = x_1 + 2", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 80, 30, "最大化"),        # 日本語ラベル
            _make_word_dict(90, 50, 200, 90, "z = x_1 + 2"), # 数式部分
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 200, 100), type="isolated", latex=r"{\sharp\mathcal{X}}", score=0.7),
        ]
        detector = FakeMathDetector(detections=detections)
        recognizer = FakeMathRecognizer(latex=r"z = x_1 + 2")
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=recognizer)

        # recognizer が呼ばれて差し替えられた LaTeX が登録されていること
        assert len(recognizer.called_paths) == 1
        assert r"z = x_1 + 2" in result.items.values()
        # garbled LaTeX は登録されていないこと
        assert r"{\sharp\mathcal{X}}" not in result.items.values()

    def test_全領域が日本語wordの場合はMathOverlayに登録しない(
        self, tmp_path: Path, mocker: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 200, 100], "contents": "制約条件", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 100, 50, "制約条件"),  # 日本語のみ
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 200, 100), type="isolated", latex=r"{\sharp}", score=0.6),
        ]
        detector = FakeMathDetector(detections=detections)
        recognizer = FakeMathRecognizer(latex=r"\text{should not appear}")
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING):
            result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=recognizer)

        # MathOverlay は空（yomitoku OCR テキストにフォールバック）
        assert result.items == {}
        # recognizer は呼ばれない
        assert len(recognizer.called_paths) == 0
        # warning ログが出ること
        assert any("スキップ" in r.message for r in caplog.records)

    def test_日本語wordなしの場合はrecognizerを呼ばない(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 200, 100], "contents": r"x^2 + y^2", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 200, 100, "x^2 + y^2"),  # 日本語なし
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 200, 100), type="isolated", latex=r"x^2 + y^2", score=0.95),
        ]
        detector = FakeMathDetector(detections=detections)
        recognizer = FakeMathRecognizer(latex=r"\should_not_be_called")
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=recognizer)

        # recognizer は呼ばれない（トリミング不要）
        assert len(recognizer.called_paths) == 0
        # 元の latex がそのまま登録される
        assert r"x^2 + y^2" in result.items.values()

    def test_再認識が失敗した場合は元のlatexを使用する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        paragraphs = [
            {"box": [0, 0, 200, 100], "contents": "最大化 z = x", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 80, 30, "最大化"),
            _make_word_dict(90, 50, 200, 90, "z = x"),
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 200, 100), type="isolated", latex=r"fallback_latex", score=0.8),
        ]
        detector = FakeMathDetector(detections=detections)
        recognizer = FakeFailingRecognizer()
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=recognizer)

        # 再認識失敗時は元の detection.latex（fallback_latex）が登録される
        assert r"fallback_latex" in result.items.values()
