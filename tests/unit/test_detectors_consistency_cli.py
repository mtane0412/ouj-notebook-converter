"""仕様: `python -m ouj_notebook_converter.detectors.consistency` の CLI のユニットテスト。

評価対象のディレクトリ全ページから誤字候補を検出して表示し、
評価セットを指定した場合は評価セットのページで適合率・再現率を計算することをテストする。
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.detectors.consistency import app

_INDEX_TERMS = (
    "余り",
    "一般角",
    "一般項",
    "因数",
    "因数分解",
    "演算",
    "円周",
    "加法",
    "階差",
    "外角",
)


def _write_cache_page(pred_dir: Path, page: int, markdown: str) -> None:
    (pred_dir / f"page_{page:04d}").mkdir(parents=True)
    (pred_dir / f"page_{page:04d}" / "raw.md").write_text(markdown, encoding="utf-8")


def _make_dirs(tmp_path: Path, *, with_index: bool) -> tuple[Path, Path]:
    """奥付の著者名を誤読したページ（p.3）を含む評価対象と、その正解を作る。"""
    pred_dir = tmp_path / "cache"
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    _write_cache_page(pred_dir, 1, "著者の隈部正博です。\n" * 4)
    _write_cache_page(pred_dir, 3, "©2018　服部正博\n")
    if with_index:
        index_lines = [f"{term}　{i + 10}  " for i, term in enumerate(_INDEX_TERMS)]
        _write_cache_page(pred_dir, 4, "# 索引\n\n" + "\n".join(index_lines))
    (truth_dir / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 3, "category": "目次・索引・前後付"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (truth_dir / "page_0003.md").write_text("©2018　隈部正博\n", encoding="utf-8")
    return pred_dir, truth_dir


def test_reports_candidates_and_scores_as_json(tmp_path: Path) -> None:
    """著者名の誤読を候補にし、評価セットで適合率・再現率を計算して JSON に書き出す。"""
    pred_dir, truth_dir = _make_dirs(tmp_path, with_index=True)
    json_path = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        ["--pred", str(pred_dir), "--truth", str(truth_dir), "--json", str(json_path)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["index_pages"] == [4]
    assert [(c["page"], c["surface"], c["suggestion"]) for c in data["candidates"]] == [
        (3, "服部正博", "隈部正博")
    ]
    assert data["scores"]["all"]["precision"] == 1.0
    assert data["scores"]["all"]["recall"] == 1.0


def test_fails_fast_without_index_pages(tmp_path: Path) -> None:
    """索引ページが無い本は、--skip-index を指定しない限り推測せずエラーにする。"""
    pred_dir, _ = _make_dirs(tmp_path, with_index=False)
    failed = CliRunner().invoke(app, ["--pred", str(pred_dir)])
    assert failed.exit_code == 1
    assert "索引ページ" in failed.output
    skipped = CliRunner().invoke(app, ["--pred", str(pred_dir), "--skip-index"])
    assert skipped.exit_code == 0, skipped.output


def test_fails_when_evaluation_page_is_missing(tmp_path: Path) -> None:
    """評価セットのページが評価対象に無ければエラーにする。"""
    pred_dir, truth_dir = _make_dirs(tmp_path, with_index=True)
    (pred_dir / "page_0003" / "raw.md").unlink()
    (pred_dir / "page_0003").rmdir()
    result = CliRunner().invoke(app, ["--pred", str(pred_dir), "--truth", str(truth_dir)])
    assert result.exit_code == 1
