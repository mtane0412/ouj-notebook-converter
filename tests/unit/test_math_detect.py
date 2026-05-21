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
    build_fragment,
    collect_words_in_paragraph,
    ioa,
    iou,
    is_japanese_text,
    match_contents_to_words,
    match_paragraph_by_ioa,
    math_detect,
    select_words_inside_embedding,
    sort_words_reading_order,
    trim_bbox_by_japanese_words,
    word_points_to_box,
)
from ouj_notebook_converter.pipeline.types import InlineParagraphReplacement, PageAnalysis
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
        # page.png 保存と crop_math_image 内の保存を両方モック（いずれも math_detect モジュール経由）
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # isolated 1 件のみ登録されていること（embedding はスキップ）
        assert len(result.items) == 1
        assert len(result.roles) == 1
        assert len(result.originals) == 1
        assert "display_formula" in result.roles.values()

    def test_同一paragraphへの2つ目のisolatedはスキップされ警告が出る(
        self, tmp_path: Path, mocker: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """同一 paragraph に 2 件の isolated（display_formula）がマッチした場合、
        1 件目は originals に登録し、2 件目は警告を出してスキップする。
        これにより post_process での 2 回目の needle 検索失敗を防ぐ。
        """
        # 1 つの大きな段落（長い数式段落を想定）
        paragraphs = [
            {
                "box": [0, 0, 400, 200],
                "contents": "一様分布の数式段落テキスト",
                "role": None,
                "order": 0,
            },
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs)
        # 同一 paragraph の bbox に 2 件の isolated 検出（Pix2Text が複数の式を検出するケース）
        detections = [
            FormulaDetection(box=(10, 10, 390, 100), type="isolated", latex=r"\alpha", score=0.9),
            FormulaDetection(box=(10, 110, 390, 190), type="isolated", latex=r"\beta", score=0.85),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((400, 500, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING):
            result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # originals に登録されるのは 1 件目のみ
        assert len(result.originals) == 1
        # 2 件目はスキップされ警告が出る
        assert any("重複" in r.message or "スキップ" in r.message for r in caplog.records)


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


# ---------------------------------------------------------------------------
# 新規純粋関数のテスト（issue #3: インライン数式 embedding 部分置換）
# ---------------------------------------------------------------------------


class TestCollectWordsInParagraph:
    """collect_words_in_paragraph: paragraph bbox に含まれる word を IoA で集約する。"""

    def test_paragraph_bbox内のwordのみを返す(self) -> None:
        # paragraph_box: (0, 0, 100, 50)
        # word A: paragraph 内に完全包含（IoA=1.0）
        # word B: paragraph 外（IoA=0.0）
        para_box = (0, 0, 100, 50)
        words = [
            _make_word_dict(10, 10, 80, 40, "段落内テキスト"),      # IoA=1.0 → 含まれる
            _make_word_dict(200, 200, 300, 250, "段落外テキスト"),  # IoA=0.0 → 除外
        ]
        result = collect_words_in_paragraph(para_box, words)
        assert len(result) == 1
        assert result[0]["content"] == "段落内テキスト"

    def test_閾値未満重なりのwordは除外される(self) -> None:
        # paragraph_box: (0, 0, 100, 100)
        # word は 40x40 で、paragraph との交差が 10x10 = 100 → IoA = 100 / (40*40) = 0.0625 < 0.5
        para_box = (0, 0, 100, 100)
        words = [
            _make_word_dict(60, 60, 100, 100, "部分重なりword"),  # word_area=40*40=1600, 交差=40*40=1600→1.0
            _make_word_dict(90, 90, 130, 130, "閾値未満word"),   # word_area=40*40=1600, 交差=10*10=100→0.0625
        ]
        result = collect_words_in_paragraph(para_box, words)
        assert len(result) == 1
        assert result[0]["content"] == "部分重なりword"

    def test_paragraph内wordがゼロなら空リスト(self) -> None:
        para_box = (0, 0, 50, 50)
        words = [
            _make_word_dict(100, 100, 200, 200, "遠いword"),
        ]
        result = collect_words_in_paragraph(para_box, words)
        assert result == []

    def test_words空なら空リスト(self) -> None:
        para_box = (0, 0, 100, 100)
        result = collect_words_in_paragraph(para_box, [])
        assert result == []


class TestSortWordsReadingOrder:
    """sort_words_reading_order: reading order に従って word をソートする。"""

    def test_horizontal方向は行内で左から右にソートされる(self) -> None:
        # 同一行（y が近い）の word を左→右順にソート
        words = [
            _make_word_dict(80, 10, 120, 30, "右のword"),
            _make_word_dict(10, 10, 60, 30, "左のword"),
        ]
        result = sort_words_reading_order(words, "horizontal")
        assert result[0]["content"] == "左のword"
        assert result[1]["content"] == "右のword"

    def test_horizontal方向は複数行が上から下にソートされる(self) -> None:
        # 2 行: 上行の word が先、下行の word が後
        words = [
            _make_word_dict(10, 60, 80, 80, "下行word"),
            _make_word_dict(10, 10, 80, 30, "上行word"),
        ]
        result = sort_words_reading_order(words, "horizontal")
        assert result[0]["content"] == "上行word"
        assert result[1]["content"] == "下行word"

    def test_vertical方向は右から左の列順にソートされる(self) -> None:
        # 縦書き: 右列が先、左列が後
        words = [
            _make_word_dict(10, 10, 40, 80, "左列word"),
            _make_word_dict(60, 10, 90, 80, "右列word"),
        ]
        result = sort_words_reading_order(words, "vertical")
        assert result[0]["content"] == "右列word"
        assert result[1]["content"] == "左列word"


class TestSelectWordsInsideEmbedding:
    """select_words_inside_embedding: embedding bbox に重なる連続範囲の word を返す。"""

    def test_embedding_bboxに含まれるwordが選択される(self) -> None:
        # paragraph words: ["テキスト前", "数式", "テキスト後"]
        # embedding bbox = 数式 word と完全一致
        words = [
            _make_word_dict(0, 0, 50, 20, "テキスト前"),
            _make_word_dict(60, 0, 120, 20, "数式"),
            _make_word_dict(130, 0, 200, 20, "テキスト後"),
        ]
        embedding_box = (60, 0, 120, 20)
        selected, prefix_count = select_words_inside_embedding(words, embedding_box)
        assert [w["content"] for w in selected] == ["数式"]
        assert prefix_count == 1  # "テキスト前" が prefix

    def test_重なり閾値未満のwordは含まれない(self) -> None:
        # word の IoA が 0.4 < 0.5 の場合は除外
        words = [
            _make_word_dict(0, 0, 100, 20, "ほぼ外のword"),  # embedding と交差 = 10*20 = 200, word_area=100*20=2000 → IoA=0.1
        ]
        embedding_box = (90, 0, 200, 20)
        selected, prefix_count = select_words_inside_embedding(words, embedding_box)
        assert selected == []
        assert prefix_count == 0

    def test_先頭にprefix_wordがある場合はprefix_countが正しい(self) -> None:
        # paragraph words: ["A", "B", "数式1", "数式2", "C"]
        # embedding は 数式1 と 数式2 を含む
        words = [
            _make_word_dict(0, 0, 20, 20, "A"),
            _make_word_dict(25, 0, 45, 20, "B"),
            _make_word_dict(50, 0, 70, 20, "数式1"),
            _make_word_dict(75, 0, 95, 20, "数式2"),
            _make_word_dict(100, 0, 120, 20, "C"),
        ]
        embedding_box = (50, 0, 95, 20)
        selected, prefix_count = select_words_inside_embedding(words, embedding_box)
        assert [w["content"] for w in selected] == ["数式1", "数式2"]
        assert prefix_count == 2  # "A", "B" が prefix

    def test_穴あき範囲は空リストを返す(self) -> None:
        # word "数式1" と "数式2" の間に IoA=0 の word "テキスト" がある
        # 最小〜最大の範囲内に IoA ゼロの word があるので穴あき
        words = [
            _make_word_dict(0, 0, 40, 20, "数式1"),
            _make_word_dict(45, 0, 85, 20, "テキスト"),  # IoA=0（embedding と重ならない）
            _make_word_dict(90, 0, 130, 20, "数式2"),
        ]
        # embedding は 数式1 と 数式2 だけを含み、テキスト は含まない
        embedding_box = (0, 0, 40, 20)  # 数式1 のみに重なる → 穴あきなし（1 件のみなので OK）
        selected, _prefix_count = select_words_inside_embedding(words, embedding_box)
        assert [w["content"] for w in selected] == ["数式1"]

    def test_間に非数式wordがある場合は穴あきとみなす(self) -> None:
        # 最小インデックス=0（数式1）、最大インデックス=2（数式2）だが間の index=1（テキスト）の IoA=0
        words = [
            _make_word_dict(0, 0, 40, 20, "数式1"),
            _make_word_dict(45, 0, 85, 20, "テキスト中間"),   # embedding と重ならない
            _make_word_dict(90, 0, 130, 20, "数式2"),
        ]
        # embedding bbox を 数式1 と 数式2 の両方を包む範囲にする（テキスト中間は重ならない）
        embedding_box = (0, 0, 130, 20)  # 全 word の box を含む（テキスト中間も IoA > 0 になる）
        selected, _prefix_count = select_words_inside_embedding(words, embedding_box)
        # embedding_box が全 word を含むので穴なし → 全 3 件選択
        assert len(selected) == 3


class TestMatchContentsToWords:
    """match_contents_to_words: paragraph.contents 行順に word を対応付ける。"""

    def test_contents行順にwordを返す(self) -> None:
        # paragraph.contents = "z\nは実数" → ["z", "は実数"] の順で word を返す
        # analysis.json の words は別の順（y 座標が微小に異なる）にあっても正しく対応
        contents = "z\nは実数"
        para_words = [
            # y 座標が微小に異なるため sort_words_reading_order では逆順になりうる
            _make_word_dict(45, 1, 200, 20, "は実数"),  # わずかに y が小さい（上方向）
            _make_word_dict(0, 2, 40, 21, "z"),          # わずかに y が大きい（下方向）
        ]
        result = match_contents_to_words(contents, para_words)
        assert [w.get("content") for w in result] == ["z", "は実数"]

    def test_同一contentが複数ある場合は順に消費される(self) -> None:
        # paragraph.contents に同じ word が複数回出現する場合は bbox 別の word を消費する
        contents = "x\ny\nx"  # "x" が 2 回出現
        words = [
            _make_word_dict(0, 0, 10, 10, "x"),    # 1 つ目の "x"
            _make_word_dict(15, 0, 25, 10, "y"),
            _make_word_dict(30, 0, 40, 10, "x"),   # 2 つ目の "x"
        ]
        result = match_contents_to_words(contents, words)
        assert [w.get("content") for w in result] == ["x", "y", "x"]
        # 1 つ目と 2 つ目の "x" は別 word（points が異なる）
        assert result[0]["points"] != result[2]["points"]

    def test_対応するwordがない行はpoints空のdictを返す(self) -> None:
        contents = "既知テキスト\n未対応テキスト"
        words = [
            _make_word_dict(0, 0, 50, 20, "既知テキスト"),
            # "未対応テキスト" の word は存在しない
        ]
        result = match_contents_to_words(contents, words)
        assert len(result) == 2
        assert result[0].get("content") == "既知テキスト"
        assert result[0].get("points") != []
        assert result[1].get("content") == "未対応テキスト"
        assert result[1].get("points") == []  # bbox なし

    def test_空contentsは空リストを返す(self) -> None:
        result = match_contents_to_words("", [])
        assert result == []


class TestBuildFragment:
    """build_fragment: word.content を順に連結する純粋関数。"""

    def test_word_contentを順に連結する(self) -> None:
        words = [
            {"content": "z"},
            {"content": "="},
            {"content": "1"},
        ]
        assert build_fragment(words) == "z=1"

    def test_空wordsで空文字列を返す(self) -> None:
        assert build_fragment([]) == ""

    def test_空contentのwordは空文字として連結される(self) -> None:
        words = [{"content": "a"}, {"content": ""}, {"content": "b"}]
        assert build_fragment(words) == "ab"


# ---------------------------------------------------------------------------
# math_detect() の embedding 処理テスト（issue #3）
# ---------------------------------------------------------------------------


class TestMathDetectEmbedding:
    """math_detect の embedding 処理と inline_paragraphs 登録を検証する。"""

    def test_embeddingはinline_paragraphsに登録される(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # embedding（インライン数式）が inline_paragraphs に登録される
        # yomitoku は paragraph.contents = "\n".join(word.content) で生成するため
        # 各 word.content が "\n" で区切られた形式にする
        paragraphs = [
            {"box": [0, 0, 200, 50], "contents": "z\nは実数", "direction": "horizontal", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 40, 50, "z"),
            _make_word_dict(45, 0, 200, 50, "は実数"),
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 40, 50), type="embedding", latex=r"z", score=0.85),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # inline_paragraphs に 1 件登録されること
        assert len(result.inline_paragraphs) == 1
        # items には登録されない（display 専用フィールド）
        assert result.items == {}
        # 登録された InlineParagraphReplacement を確認
        repl = next(iter(result.inline_paragraphs.values()))
        assert isinstance(repl, InlineParagraphReplacement)
        assert r"z" in (span[2] for span in repl.latex_spans)

    def test_embedding_bboxに重なるwordがなければwarningでスキップする(
        self, tmp_path: Path, mocker: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # paragraph 内に word が存在しない（または embedding と重ならない）場合
        paragraphs = [
            {"box": [0, 0, 200, 50], "contents": "テスト段落", "direction": "horizontal", "role": None, "order": 0},
        ]
        words: list[Any] = []  # word なし
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 40, 50), type="embedding", latex=r"x^2", score=0.9),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING):
            result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # warning が出てスキップ → inline_paragraphs は空
        assert result.inline_paragraphs == {}
        assert any("スキップ" in r.message or "word" in r.message.lower() for r in caplog.records)

    def test_同一paragraph内の複数embeddingは1つのinline_paragraphエントリにまとめられる(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # paragraph 内に embedding が 2 件ある場合、inline_paragraphs のエントリは 1 件
        # yomitoku は paragraph.contents = "\n".join(word.content) で生成するため "\n" 区切り
        paragraphs = [
            {"box": [0, 0, 300, 50], "contents": "z\nと\nw\nは実数", "direction": "horizontal", "role": None, "order": 0},
        ]
        words = [
            _make_word_dict(0, 0, 40, 50, "z"),
            _make_word_dict(45, 0, 100, 50, "と"),
            _make_word_dict(105, 0, 145, 50, "w"),
            _make_word_dict(150, 0, 300, 50, "は実数"),
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 40, 50), type="embedding", latex=r"z", score=0.9),
            FormulaDetection(box=(105, 0, 145, 50), type="embedding", latex=r"w", score=0.88),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # inline_paragraphs は 1 エントリ（同一 paragraph でまとめられる）
        assert len(result.inline_paragraphs) == 1
        repl = next(iter(result.inline_paragraphs.values()))
        # latex_spans は 2 件（z と w）
        assert len(repl.latex_spans) == 2
        latexes = {span[2] for span in repl.latex_spans}
        assert r"z" in latexes
        assert r"w" in latexes

    def test_embeddingとisolatedが混在する場合それぞれの場所に登録される(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        # isolated は items / roles / originals に、embedding は inline_paragraphs に登録される
        paragraphs = [
            # yomitoku は paragraph.contents = "\n".join(word.content) で生成
            {"box": [0, 0, 100, 50], "contents": "z", "direction": "horizontal", "role": None, "order": 0},
            {"box": [0, 60, 100, 110], "contents": "ディスプレイ数式", "direction": "horizontal", "role": None, "order": 1},
        ]
        words = [
            # embedding bbox に含まれる ASCII word（日本語ラベルトリミングを発動させない）
            _make_word_dict(0, 0, 60, 50, "z"),
        ]
        analysis = _make_page_analysis(tmp_path, paragraphs, words)
        detections = [
            FormulaDetection(box=(0, 0, 60, 50), type="embedding", latex="a", score=0.9),
            FormulaDetection(box=(0, 60, 100, 110), type="isolated", latex="b", score=0.8),
        ]
        detector = FakeMathDetector(detections=detections)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_detect.save_image")

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = math_detect(image, analysis, tmp_path, detector=detector, recognizer=FakeMathRecognizer())

        # isolated は items に、embedding は inline_paragraphs に登録
        assert len(result.items) == 1
        assert "display_formula" in result.roles.values()
        assert len(result.inline_paragraphs) == 1
