"""仕様: `python -m ouj_notebook_converter.detectors` の CLI のユニットテスト。

評価対象のディレクトリ全ページの数式を検算して誤読候補を表示し、
評価セットを指定した場合は評価セットのページで適合率・再現率を計算することをテストする。
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ouj_notebook_converter.detectors import cli


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """2 ページ分の評価対象（ページキャッシュ）と、そのうち 1 ページ分の評価セットを作る。"""
    pred_dir = tmp_path / "cache"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    # p.70: 4 乗根を 3 乗根と誤読した等式（検算で検出できる誤読）
    (pred_dir / "page_0070").mkdir(parents=True)
    (pred_dir / "page_0070" / "raw.md").write_text(
        "$$a = (\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4$$\n", encoding="utf-8"
    )
    # p.71: 誤読の無いページ（評価セットには含めない）
    (pred_dir / "page_0071").mkdir()
    (pred_dir / "page_0071" / "raw.md").write_text("$(x+1)^2 = x^2 + 2x + 1$\n", encoding="utf-8")
    (truth_dir / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 70, "category": "数式中心"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (truth_dir / "page_0070.md").write_text(
        "$$a = (\\sqrt[3]{a})^3 = (\\sqrt[4]{a})^4$$\n", encoding="utf-8"
    )
    return pred_dir, truth_dir


def test_reports_violations_and_scores_as_json(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """全ページの誤読候補と、評価セットのページでの適合率・再現率を JSON に書き出す。"""
    pred_dir, truth_dir = dirs
    json_path = tmp_path / "result.json"

    result = CliRunner().invoke(
        cli.app,
        ["--pred", str(pred_dir), "--truth", str(truth_dir), "--json", str(json_path)],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["page_count"] == 2
    assert data["status_counts"]["violated"] == 1
    assert data["status_counts"]["holds"] == 2
    assert [(v["page"], v["rhs"]) for v in data["violations"]] == [(70, "(\\sqrt[3]{a})^4")]
    assert data["scores"]["total"] == {
        "flagged": 1,
        "true_positive": 1,
        "misread": 1,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert [p["page"] for p in data["scores"]["pages"]] == [70]


def test_missing_pred_dir_fails(tmp_path: Path) -> None:
    """評価対象のディレクトリが無ければ、エラーを表示して終了コード 1 で終わる。"""
    result = CliRunner().invoke(cli.app, ["--pred", str(tmp_path / "なし")])
    assert result.exit_code == 1
    assert "見つかりません" in result.output
