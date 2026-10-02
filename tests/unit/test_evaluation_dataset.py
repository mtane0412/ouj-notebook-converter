"""仕様: evaluation.dataset モジュール（評価セットと評価対象出力の読み込み）のユニットテスト。

評価セット（manifest.json と正解 page_NNNN.md）の読み込みと、評価対象の出力ディレクトリ
（--no-combine の出力 または ページキャッシュ）からのページ Markdown 読み込みをテストする。
"""

import json
from pathlib import Path

import pytest

from ouj_notebook_converter.evaluation.dataset import (
    EvalPage,
    list_prediction_pages,
    load_manifest,
    read_prediction,
    read_truth,
)


def _write_manifest(truth_dir: Path, pages: list[dict[str, object]]) -> None:
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "manifest.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False), encoding="utf-8"
    )


class TestLoadManifest:
    """manifest.json の読み込み。"""

    def test_reads_pages_and_categories(self, tmp_path: Path) -> None:
        """評価ページの番号と層（カテゴリ）を読み込む。"""
        _write_manifest(
            tmp_path,
            [{"page": 70, "category": "数式中心"}, {"page": 5, "category": "目次・索引"}],
        )

        assert load_manifest(tmp_path) == [
            EvalPage(page=70, category="数式中心"),
            EvalPage(page=5, category="目次・索引"),
        ]

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """manifest.json が無ければ FileNotFoundError を送出する。"""
        with pytest.raises(FileNotFoundError, match=r"manifest\.json"):
            load_manifest(tmp_path)

    def test_duplicate_page_raises(self, tmp_path: Path) -> None:
        """同じページ番号が重複していれば ValueError を送出する。"""
        _write_manifest(
            tmp_path,
            [{"page": 70, "category": "数式中心"}, {"page": 70, "category": "地の文中心"}],
        )

        with pytest.raises(ValueError, match="70"):
            load_manifest(tmp_path)

    def test_invalid_entry_raises(self, tmp_path: Path) -> None:
        """ページ番号が正の整数でなければ ValueError を送出する。"""
        _write_manifest(tmp_path, [{"page": 0, "category": "数式中心"}])

        with pytest.raises(ValueError):
            load_manifest(tmp_path)


class TestReadTruth:
    """正解 Markdown の読み込み。"""

    def test_reads_page_file(self, tmp_path: Path) -> None:
        """page_NNNN.md を読み込む。"""
        (tmp_path / "page_0070.md").write_text("正解の本文", encoding="utf-8")

        assert read_truth(tmp_path, 70) == "正解の本文"

    def test_missing_page_raises(self, tmp_path: Path) -> None:
        """正解ファイルが無ければ FileNotFoundError を送出する。"""
        with pytest.raises(FileNotFoundError, match=r"page_0070\.md"):
            read_truth(tmp_path, 70)


class TestReadPrediction:
    """評価対象の出力の読み込み。"""

    def test_reads_no_combine_output(self, tmp_path: Path) -> None:
        """--no-combine 出力の page_NNNN.md はそのまま読み込む。"""
        (tmp_path / "page_0070.md").write_text("### 4.3 累乗根", encoding="utf-8")

        assert read_prediction(tmp_path, 70) == "### 4.3 累乗根"

    def test_reads_cache_with_cleanup_applied(self, tmp_path: Path) -> None:
        """キャッシュの raw.md は変換時と同じ後処理（見出しレベルの正規化など）を適用して読み込む。"""
        page_dir = tmp_path / "page_0070"
        page_dir.mkdir()
        (page_dir / "raw.md").write_text("### 4.3 累乗根", encoding="utf-8")

        assert read_prediction(tmp_path, 70) == "## 4.3 累乗根"

    def test_ambiguous_layout_raises(self, tmp_path: Path) -> None:
        """両方の形式が存在すればどちらを評価すべきか決められないため ValueError を送出する。"""
        (tmp_path / "page_0070.md").write_text("出力", encoding="utf-8")
        (tmp_path / "page_0070").mkdir()
        (tmp_path / "page_0070" / "raw.md").write_text("キャッシュ", encoding="utf-8")

        with pytest.raises(ValueError, match="page_0070"):
            read_prediction(tmp_path, 70)

    def test_missing_page_raises(self, tmp_path: Path) -> None:
        """どちらの形式も無ければ FileNotFoundError を送出する。"""
        with pytest.raises(FileNotFoundError, match="page_0070"):
            read_prediction(tmp_path, 70)


class TestListPredictionPages:
    """list_prediction_pages: 評価対象ディレクトリにあるページ番号の一覧。"""

    def test_lists_pages_of_both_layouts_in_order(self, tmp_path: Path) -> None:
        (tmp_path / "page_0012.md").write_text("出力", encoding="utf-8")
        (tmp_path / "page_0003").mkdir()
        (tmp_path / "page_0003" / "raw.md").write_text("キャッシュ", encoding="utf-8")
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        assert list_prediction_pages(tmp_path) == [3, 12]

    def test_ignores_cache_dir_without_raw_md(self, tmp_path: Path) -> None:
        (tmp_path / "page_0005").mkdir()
        assert list_prediction_pages(tmp_path) == []

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_prediction_pages(tmp_path / "なし")
