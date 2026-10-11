"""仕様: evaluation.text_layer_cer（searchable PDF のテキストレイヤーの CER 評価、issue #42）のテスト。

評価セットのページについて、yomitoku の文字だけのテキストレイヤーと、Gemini の文字を差し込んだ
テキストレイヤーの searchable PDF を作り、PDF から抽出したテキストを正解と比べる。
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.evaluation.text_layer_cer import app, japanese_only, score_text_layer


class TestScoreTextLayer:
    """抽出したテキストと正解 Markdown の比較。"""

    def test_full_text_compares_with_flattened_truth(self) -> None:
        """全文の CER は、正解の Markdown を記法を除き数式を平文化した文字列と比べる。"""
        score = score_text_layer("ここで $x^2 = 4$ とおく。", "ここで x2=4 とおく。\r\n")

        assert score.full.edits == 0
        assert score.full.truth_length == len("ここでx2=4とおく。")

    def test_japanese_only_ignores_math_and_symbols(self) -> None:
        """日本語の CER は、かな・漢字だけを比べる（数式の書き方の違いを数えない）。"""
        score = score_text_layer("ここで $x^2 = 4$ とおく。", "ここで x^2=4 とおく")

        assert score.japanese.edits == 0
        assert score.japanese.truth_length == len("ここでとおく")

    def test_counts_misread_characters(self) -> None:
        score = score_text_layer("数の概念", "数の慨念")

        assert score.full.edits == 1
        assert score.japanese.edits == 1


def test_japanese_only_keeps_kana_kanji_and_long_vowel() -> None:
    assert japanese_only("1.1 データ（A）の値は々") == "データの値は々"


def _make_source_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas

    c = rl_canvas.Canvas(str(path), pagesize=A4)
    c.showPage()
    c.save()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_cli_compares_yomitoku_and_gemini_text_layers(tmp_path: Path) -> None:
    """yomitoku が誤読した文字を、Gemini の文字を差し込んだテキストレイヤーでは正しく抽出できる。"""
    source_pdf = tmp_path / "教材.pdf"
    _make_source_pdf(source_pdf)

    truth_dir = tmp_path / "eval"
    _write_json(truth_dir / "manifest.json", {"pages": [{"page": 1, "category": "地の文中心"}]})
    (truth_dir / "page_0001.md").write_text("数の概念について", encoding="utf-8")

    yomitoku_dir = tmp_path / "yomitoku"
    _write_json(
        yomitoku_dir / "page_0001" / "analysis.json",
        {
            "paragraphs": [{"box": [0, 0, 1600, 2000], "order": 0, "direction": "horizontal"}],
            "figures": [],
            "words": [
                {
                    "content": "数の慨念について",
                    "direction": "horizontal",
                    "points": [[100, 200], [900, 200], [900, 240], [100, 240]],
                }
            ],
        },
    )

    gemini_dir = tmp_path / "gemini"
    (gemini_dir / "page_0001").mkdir(parents=True)
    (gemini_dir / "page_0001" / "raw.md").write_text("数の概念にっいて", encoding="utf-8")
    _write_json(gemini_dir / "page_0001" / "analysis.json", {"backend": "gemini", "words": []})
    corrections = tmp_path / "corrections.json"
    _write_json(
        corrections,
        {
            "version": 1,
            "corrections": [
                {"page": 1, "before": "にっいて", "after": "について"},
                {"page": 5, "before": "評価対象外の誤読", "after": "評価対象外の修正"},
            ],
        },
    )
    result_json = tmp_path / "result.json"

    result = CliRunner().invoke(
        app,
        [
            "--truth",
            str(truth_dir),
            "--pdf",
            str(source_pdf),
            "--word-bbox-cache",
            str(yomitoku_dir),
            "--gemini-cache",
            str(gemini_dir),
            "--corrections",
            str(corrections),
            "--out-dir",
            str(tmp_path / "out"),
            "--json",
            str(result_json),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result_json.read_text(encoding="utf-8"))
    assert report["total"]["yomitoku"]["full"]["edits"] == 1
    assert report["total"]["gemini"]["full"]["edits"] == 0
    assert report["pages"][0]["page"] == 1
    assert report["alignment"]["chars_replaced"] == 1
    assert (tmp_path / "out" / "gemini.pdf").is_file()
    # 評価セットに無いページの修正は、無視せず適用しなかったページとして報告する
    assert report["corrections_outside_eval"] == [5]
    assert "評価対象外のため適用しなかった修正のページ: 5" in result.output


def test_cli_scores_existing_searchable_pdf_by_book_page(tmp_path: Path) -> None:
    """--legacy-pdf を渡すと、既存の searchable PDF の該当ページ（書籍のページ番号）も評価する。"""
    from ouj_notebook_converter.exporters.pdf import export_pdf
    from ouj_notebook_converter.pipeline.types import PageMarkdown

    source_pdf = tmp_path / "教材.pdf"
    _make_source_pdf(source_pdf)
    truth_dir = tmp_path / "eval"
    _write_json(truth_dir / "manifest.json", {"pages": [{"page": 1, "category": "地の文中心"}]})
    (truth_dir / "page_0001.md").write_text("本文", encoding="utf-8")
    yomitoku_dir = tmp_path / "yomitoku"
    words_json = yomitoku_dir / "page_0001" / "analysis.json"
    _write_json(
        words_json,
        {
            "paragraphs": [],
            "figures": [],
            "words": [
                {"content": "本文", "points": [[100, 200], [300, 200], [300, 240], [100, 240]]}
            ],
        },
    )
    gemini_dir = tmp_path / "gemini"
    (gemini_dir / "page_0001").mkdir(parents=True)
    (gemini_dir / "page_0001" / "raw.md").write_text("本文", encoding="utf-8")
    legacy_pdf = tmp_path / "既存.pdf"
    export_pdf(
        [PageMarkdown(page_index=0, markdown="", yomitoku_json_path=words_json)],
        legacy_pdf,
        tmp_path / "assets",
        source_pdf=source_pdf,
    )
    result_json = tmp_path / "result.json"

    result = CliRunner().invoke(
        app,
        [
            "--truth",
            str(truth_dir),
            "--pdf",
            str(source_pdf),
            "--word-bbox-cache",
            str(yomitoku_dir),
            "--gemini-cache",
            str(gemini_dir),
            "--legacy-pdf",
            str(legacy_pdf),
            "--out-dir",
            str(tmp_path / "out"),
            "--json",
            str(result_json),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result_json.read_text(encoding="utf-8"))
    assert report["total"]["legacy"]["full"]["edits"] == 0
