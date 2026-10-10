"""仕様: `python -m ouj_notebook_converter.detectors.cross_ocr_diff` の CLI のユニットテスト。

Gemini と yomitoku のキャッシュを比較して誤読候補を JSON に書き出し、
評価セットを指定した場合は適合率・再現率を計算することをテストする。
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ouj_notebook_converter.detectors.cross_ocr_diff import cli


def _write_analysis(directory: Path, text: str) -> None:
    """1 段落だけの yomitoku の analysis.json を書く。"""
    directory.mkdir(parents=True)
    analysis = {
        "paragraphs": [{"box": [100, 300, 900, 340], "contents": text, "role": None, "order": 0}],
        "figures": [],
        "tables": [],
        "words": [],
    }
    (directory / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False))


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Gemini・yomitoku のキャッシュと、評価セットを作る。

    p.3 は Gemini が「服部」と誤読したページ、p.4 は食い違いの無いページ。
    """
    gemini_dir = tmp_path / "gemini"
    yomitoku_dir = tmp_path / "yomitoku"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    for page, gemini_text, yomitoku_text in [
        (3, "著者は服部正博です。", "著者は隈部正博です。"),
        (4, "有理数を小数で表します。", "有理数を小数で表します。"),
    ]:
        (gemini_dir / f"page_{page:04d}").mkdir(parents=True)
        (gemini_dir / f"page_{page:04d}" / "raw.md").write_text(gemini_text, encoding="utf-8")
        _write_analysis(yomitoku_dir / f"page_{page:04d}", yomitoku_text)
    (truth_dir / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 3, "category": "目次・索引・前後付"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (truth_dir / "page_0003.md").write_text("著者は隈部正博です。", encoding="utf-8")
    return gemini_dir, yomitoku_dir, truth_dir


def test_writes_candidates_and_scores_as_json(
    dirs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """誤読候補を LLM 判定用の形式で、評価セットでの適合率・再現率とともに JSON に書き出す。"""
    gemini_dir, yomitoku_dir, truth_dir = dirs
    json_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli.app,
        [
            "--gemini", str(gemini_dir),
            "--yomitoku", str(yomitoku_dir),
            "--truth", str(truth_dir),
            "--json", str(json_path),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["format_version"] == 1
    assert data["page_count"] == 2
    assert len(data["candidates"]) == 1
    candidate = data["candidates"][0]
    assert candidate["page"] == 3
    assert candidate["gemini"]["text"] == "服"
    assert candidate["yomitoku"]["text"] == "隈"
    assert candidate["yomitoku"]["bbox"] == [100, 300, 900, 340]
    assert data["scores"]["total"]["precision"] == 1.0
    assert data["scores"]["total"]["recall"] == 1.0


def test_missing_analysis_json_fails_fast(dirs: tuple[Path, Path, Path]) -> None:
    """yomitoku 側に analysis.json が無いページがあれば、黙って飛ばさずエラーにする。"""
    gemini_dir, yomitoku_dir, _ = dirs
    (yomitoku_dir / "page_0004" / "analysis.json").unlink()

    result = CliRunner().invoke(
        cli.app, ["--gemini", str(gemini_dir), "--yomitoku", str(yomitoku_dir)]
    )

    assert result.exit_code == 1
    assert "analysis.json" in result.output


def test_noise_filters_can_be_disabled(dirs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """ノイズ除去の効果を測るため、除去を個別に無効化できる。"""
    gemini_dir, yomitoku_dir, _ = dirs
    json_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli.app,
        [
            "--gemini", str(gemini_dir),
            "--yomitoku", str(yomitoku_dir),
            "--no-exclude-header",
            "--no-exclude-figures",
            "--no-normalize-variants",
            "--min-run-length", "1",
            "--json", str(json_path),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    options = json.loads(json_path.read_text(encoding="utf-8"))["options"]
    assert options["exclude_header"] is False
    assert options["exclude_figures"] is False
    assert options["normalize_variants"] is False
    assert options["min_run_length"] == 1
