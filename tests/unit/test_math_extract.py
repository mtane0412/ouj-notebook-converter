"""仕様: math_extract ステージ（数式 paragraph 抽出・クロップ・エンジン呼び出し）の動作検証。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ouj_notebook_converter.pipeline.stages.math_extract import (
    MathParagraph,
    crop_math_image,
    extract_math_paragraphs,
    math_extract,
)
from ouj_notebook_converter.pipeline.types import PageAnalysis


def _make_analysis_json(tmp_path: Path, paragraphs: list[dict[str, object]]) -> Path:
    """テスト用の analysis.json を作成するヘルパ。"""
    data = {"paragraphs": paragraphs, "tables": [], "words": [], "figures": []}
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class FakeMathEngine:
    """テスト用の固定 LaTeX を返す数式エンジン。"""

    def __init__(self, latex: str = r"\alpha + \beta") -> None:
        self._latex = latex
        self.called_paths: list[Path] = []

    def recognize(self, image_path: Path) -> str:
        self.called_paths.append(image_path)
        return self._latex


class TestExtractMathParagraphs:
    def test_formula_roleのみ抽出する(self, tmp_path: Path) -> None:
        """display_formula と inline_formula だけが抽出され、他の role は無視される。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                # 非数式 role は除外される
                {
                    "box": [0, 0, 100, 20],
                    "contents": "第1章",
                    "direction": None,
                    "order": 0,
                    "role": "section_headings",
                },
                {
                    "box": [0, 20, 100, 40],
                    "contents": "E=mc^2",
                    "direction": None,
                    "order": 1,
                    "role": "inline_formula",
                },
                {
                    "box": [0, 40, 100, 60],
                    "contents": "F_n の和",
                    "direction": None,
                    "order": 2,
                    "role": "display_formula",
                },
                {
                    "box": [0, 60, 100, 80],
                    "contents": "普通のテキスト",
                    "direction": None,
                    "order": 3,
                    "role": None,
                },
            ],
        )

        result = extract_math_paragraphs(json_path)

        assert len(result) == 2
        assert result[0].role == "inline_formula"
        assert result[1].role == "display_formula"

    def test_paragraphsインデックスが保持される(self, tmp_path: Path) -> None:
        """抽出された paragraph は元のリストのインデックスを保持する。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                {
                    "box": [0, 0, 100, 20],
                    "contents": "キャプション",
                    "direction": None,
                    "order": 0,
                    "role": "caption",
                },
                {
                    "box": [0, 20, 100, 40],
                    "contents": "積分",
                    "direction": None,
                    "order": 1,
                    "role": "display_formula",
                },
            ],
        )

        result = extract_math_paragraphs(json_path)

        # paragraphs 配列のインデックス 1（0 番目は caption で除外）
        assert result[0].index == 1

    def test_contents_Noneの数式paragraphは除外される(self, tmp_path: Path) -> None:
        """contents が None の場合は置換対象文字列がないためスキップする。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                {
                    "box": [0, 0, 100, 20],
                    "contents": None,
                    "direction": None,
                    "order": 0,
                    "role": "display_formula",
                },
                {
                    "box": [0, 20, 100, 40],
                    "contents": "x+y",
                    "direction": None,
                    "order": 1,
                    "role": "inline_formula",
                },
            ],
        )

        result = extract_math_paragraphs(json_path)

        assert len(result) == 1
        assert result[0].original_contents == "x+y"


class TestCropMathImage:
    def test_bbox範囲でクロップして連番PNGを保存する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """bbox [x1,y1,x2,y2] の範囲でクロップし、paragraph.index をゼロパディングした PNG 名で保存する。"""
        # 10x10 の黒画像（BGR 形式）
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        paragraph = MathParagraph(
            index=3, role="display_formula", box=(2, 2, 6, 7), original_contents="積分式"
        )
        output_dir = tmp_path / "math"

        mock_save = mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        saved_path = crop_math_image(image, paragraph, output_dir)

        assert saved_path == output_dir / "0003.png"
        mock_save.assert_called_once()
        # crop の shape: y2-y1=5, x2-x1=4, channels=3
        actual_crop = mock_save.call_args[0][0]
        assert actual_crop.shape == (5, 4, 3)

    def test_bbox範囲外をクランプして成功する(self, tmp_path: Path, mocker: MagicMock) -> None:
        """bbox が画像サイズを超えていても、クランプして正常に動作する。"""
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        paragraph = MathParagraph(
            index=0, role="inline_formula", box=(3, 3, 100, 100), original_contents="y"
        )
        mock_save = mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        saved_path = crop_math_image(image, paragraph, tmp_path)

        assert saved_path.name == "0000.png"
        # x1=3, y1=3 → 5x5 でクランプすると x2=5, y2=5 → shape (2, 2, 3)
        actual_crop = mock_save.call_args[0][0]
        assert actual_crop.shape == (2, 2, 3)

    def test_bboxサイズゼロでValueError(self, tmp_path: Path, mocker: MagicMock) -> None:
        """クランプ後に bbox サイズが 0 になる場合は ValueError を送出する。"""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        # x1 == x2 → 幅ゼロ
        paragraph = MathParagraph(
            index=0, role="display_formula", box=(5, 5, 5, 10), original_contents="z"
        )
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        with pytest.raises(ValueError, match="数式 bbox のサイズが 0"):
            crop_math_image(image, paragraph, tmp_path)


class TestMathExtract:
    def test_数式なしのページで空のMathOverlayを返す(self, tmp_path: Path) -> None:
        """数式 role を持つ paragraph がなければ全フィールドが空の MathOverlay を返す。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                {
                    "box": [0, 0, 100, 20],
                    "contents": "目次",
                    "direction": None,
                    "order": 0,
                    "role": "section_headings",
                }
            ],
        )
        (tmp_path / "raw.md").write_text("目次\n", encoding="utf-8")
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=json_path,
            figure_paths=[],
            markdown_raw_path=tmp_path / "raw.md",
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        overlay = math_extract(image, analysis, tmp_path, engine=FakeMathEngine())

        assert overlay.items == {}
        assert overlay.roles == {}
        assert overlay.originals == {}

    def test_engineのrecognizeを各数式に1回ずつ呼ぶ(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """数式 paragraph が N 個あれば engine.recognize が N 回呼ばれる。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                {
                    "box": [0, 0, 100, 20],
                    "contents": "分数式",
                    "direction": None,
                    "order": 0,
                    "role": "display_formula",
                },
                {
                    "box": [0, 20, 100, 40],
                    "contents": "シグマ式",
                    "direction": None,
                    "order": 1,
                    "role": "inline_formula",
                },
            ],
        )
        (tmp_path / "raw.md").write_text("", encoding="utf-8")
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=json_path,
            figure_paths=[],
            markdown_raw_path=tmp_path / "raw.md",
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        engine = FakeMathEngine(latex=r"\sum_{i=0}^{n} x_i")
        math_extract(image, analysis, tmp_path, engine=engine)

        assert len(engine.called_paths) == 2

    def test_MathOverlayにitems_roles_originalsを設定する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """engine の戻り値が items に入り、roles と originals も対応付けられる。"""
        json_path = _make_analysis_json(
            tmp_path,
            [
                {
                    "box": [10, 20, 80, 50],
                    "contents": "フーリエ変換式",
                    "direction": None,
                    "order": 0,
                    "role": "display_formula",
                }
            ],
        )
        (tmp_path / "raw.md").write_text("", encoding="utf-8")
        analysis = PageAnalysis(
            page_index=0,
            yomitoku_json_path=json_path,
            figure_paths=[],
            markdown_raw_path=tmp_path / "raw.md",
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        mocker.patch("ouj_notebook_converter.pipeline.stages.math_extract.save_image")

        engine = FakeMathEngine(latex=r"\hat{f}(\xi)")
        overlay = math_extract(image, analysis, tmp_path, engine=engine)

        assert len(overlay.items) == 1
        crop_path = next(iter(overlay.items.keys()))
        assert overlay.items[crop_path] == r"\hat{f}(\xi)"
        assert overlay.roles[crop_path] == "display_formula"
        assert overlay.originals[crop_path] == "フーリエ変換式"
