"""仕様: 人手修正ファイル（corrections）の読み込みと適用のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouj_notebook_converter.corrections import (
    Correction,
    CorrectionError,
    apply_corrections,
    load_corrections,
)


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadCorrections:
    def test_ページごとに修正を分類して読み込む(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            {
                "version": 1,
                "corrections": [
                    {"page": 3, "before": "服部正博", "after": "隈部正博"},
                    {
                        "page": 70,
                        "before": "A",
                        "after": "B",
                        "reason": "添字の誤読",
                        "reviewer": "田中",
                        "date": "2026-10-01",
                    },
                    {"page": 3, "before": "C", "after": "D"},
                ],
            },
        )
        result = load_corrections(path)
        assert [c.before for c in result[3]] == ["服部正博", "C"]
        assert result[70][0] == Correction(
            page=70, before="A", after="B", reason="添字の誤読", reviewer="田中", date="2026-10-01"
        )

    def test_ファイルが存在しない場合は例外(self, tmp_path: Path) -> None:
        with pytest.raises(CorrectionError, match="見つかりません"):
            load_corrections(tmp_path / "none.json")

    def test_JSONとして不正なら例外(self, tmp_path: Path) -> None:
        path = tmp_path / "corrections.json"
        path.write_text("{壊れた", encoding="utf-8")
        with pytest.raises(CorrectionError, match="JSON"):
            load_corrections(path)

    def test_versionが未対応なら例外(self, tmp_path: Path) -> None:
        with pytest.raises(CorrectionError, match="version"):
            load_corrections(_write(tmp_path, {"version": 2, "corrections": []}))

    @pytest.mark.parametrize(
        "entry",
        [
            {"before": "A", "after": "B"},  # page 欠落
            {"page": 0, "before": "A", "after": "B"},  # page は 1 以上
            {"page": "3", "before": "A", "after": "B"},  # page は整数
            {"page": 3, "after": "B"},  # before 欠落
            {"page": 3, "before": "", "after": "B"},  # before は空不可
            {"page": 3, "before": "A"},  # after 欠落
            {"page": 3, "before": "A", "after": "A"},  # 変更なし
            {"page": 3, "before": "A", "after": "B", "unknown": 1},  # 未知キー
        ],
    )
    def test_不正なエントリは何番目かを示して例外(self, tmp_path: Path, entry: dict) -> None:
        path = _write(tmp_path, {"version": 1, "corrections": [entry]})
        with pytest.raises(CorrectionError, match=r"corrections\[0\]"):
            load_corrections(path)


class TestApplyCorrections:
    def test_一意に一致する文字列を置換する(self) -> None:
        corrections = (Correction(page=3, before="服部", after="隈部"),)
        assert apply_corrections("©2018　服部正博", corrections) == "©2018　隈部正博"

    def test_複数の修正を順に適用する(self) -> None:
        corrections = (
            Correction(page=1, before="あ", after="い"),
            Correction(page=1, before="い。", after="う。"),
        )
        assert apply_corrections("あ。", corrections) == "う。"

    def test_修正が空なら入力をそのまま返す(self) -> None:
        assert apply_corrections("本文", ()) == "本文"

    def test_置換前が見つからなければ頁と修正を示して例外(self) -> None:
        corrections = (Correction(page=70, before="存在しない", after="X"),)
        with pytest.raises(CorrectionError, match=r"70 ページ.*見つかりません.*存在しない"):
            apply_corrections("本文", corrections)

    def test_置換前が複数一致すれば頁と件数を示して例外(self) -> None:
        corrections = (Correction(page=60, before="無限", after="無制限"),)
        with pytest.raises(CorrectionError, match=r"60 ページ.*2 か所"):
            apply_corrections("無限に続く。無限小数。", corrections)
