"""仕様: `python -m ouj_notebook_converter.detectors.multi_run_variance` の CLI のユニットテスト。

複数回の OCR 出力を比較して不確実箇所を JSON に書き出し、
評価セットを指定した場合は適合率・再現率と多数決の効果を計算することをテストする。
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.detectors.multi_run_variance import cli

RUN_TEXTS = [
    "著者は服部正博です。ここで $x^2+1$ を得る。",
    "著者は隈部正博です。ここで $x^3+1$ を得る。",
    "著者は隈部正博です。ここで $x^2+1$ を得る。",
]
TRUTH_TEXT = "著者は隈部正博です。ここで $x^2+1$ を得る。"


def _make_runs(tmp_path: Path) -> list[Path]:
    runs = []
    for i, text in enumerate(RUN_TEXTS, start=1):
        page_dir = tmp_path / f"run{i}" / "page_0003"
        page_dir.mkdir(parents=True)
        (page_dir / "raw.md").write_text(text, encoding="utf-8")
        runs.append(tmp_path / f"run{i}")
    return runs


def _make_truth(tmp_path: Path) -> Path:
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 3, "category": "目次・索引・前後付"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (truth / "page_0003.md").write_text(TRUTH_TEXT, encoding="utf-8")
    return truth


def test_writes_candidates_for_llm_judge(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path)
    json_path = tmp_path / "result.json"
    result = CliRunner().invoke(cli.app, [*map(str, runs), "--json", str(json_path)])
    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["format_version"] == 1
    assert data["run_dirs"] == [str(r) for r in runs]
    assert data["page_count"] == 1
    assert {c["granularity"] for c in data["candidates"]} == {"prose", "formula"}
    prose = next(c for c in data["candidates"] if c["granularity"] == "prose")
    assert prose["page"] == 3
    assert prose["texts"] == ["服", "隈", "隈"]
    assert "scores" not in data


def test_scores_detection_and_vote_with_truth(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path)
    truth = _make_truth(tmp_path)
    json_path = tmp_path / "result.json"
    result = CliRunner().invoke(
        cli.app, [*map(str, runs), "--truth", str(truth), "--json", str(json_path)]
    )
    assert result.exit_code == 0, result.output
    scores = json.loads(json_path.read_text(encoding="utf-8"))["scores"]
    detection = scores["detection"]["3"]
    assert detection["1"]["prose"]["recall"] == 1.0
    assert detection["1"]["formula"]["recall"] == 1.0
    vote = scores["vote"]["3"]
    assert vote["voted"]["cer"] == 0.0
    assert vote["voted"]["math_f1"] == 1.0
    assert vote["single_mean"]["cer"] > 0.0


def test_subset_sizes_limit_combinations(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path)
    truth = _make_truth(tmp_path)
    json_path = tmp_path / "result.json"
    result = CliRunner().invoke(
        cli.app,
        [*map(str, runs), "--truth", str(truth), "--subset-size", "2", "--json", str(json_path)],
    )
    assert result.exit_code == 0, result.output
    scores = json.loads(json_path.read_text(encoding="utf-8"))["scores"]
    assert list(scores["vote"]) == ["2"]


def test_fails_with_less_than_two_runs(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path)
    result = CliRunner().invoke(cli.app, [str(runs[0])])
    assert result.exit_code == 1
    assert "エラー" in result.output


def test_fails_when_a_run_lacks_the_page(tmp_path: Path) -> None:
    runs = _make_runs(tmp_path)
    truth = _make_truth(tmp_path)
    (runs[1] / "page_0003" / "raw.md").unlink()
    result = CliRunner().invoke(cli.app, [*map(str, runs), "--truth", str(truth)])
    assert result.exit_code == 1
    assert "エラー" in result.output
