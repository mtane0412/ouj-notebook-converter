"""仕様: evaluation.report モジュールと評価 CLI のユニットテスト。

評価セット全体を評価して層（カテゴリ）別・全体の指標を集計する evaluate_dataset と、
`python -m ouj_notebook_converter.evaluation` の CLI をテストする。
KaTeX 検査は外部プロセスを使うため、フェイクの検査関数に差し替える。
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ouj_notebook_converter.evaluation import cli
from ouj_notebook_converter.evaluation.markdown_parts import Formula
from ouj_notebook_converter.evaluation.report import evaluate_dataset


def _fake_katex(formulas: Sequence[Formula]) -> list[str | None]:
    """\\cline を含む数式だけを描画不能とみなすフェイクの KaTeX 検査。"""
    return ["Undefined control sequence: \\cline" if "\\cline" in f.tex else None for f in formulas]


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    """2 ページ分の評価セットと評価対象の出力を作る。"""
    truth_dir = tmp_path / "truth"
    pred_dir = tmp_path / "pred"
    truth_dir.mkdir()
    pred_dir.mkdir()
    (truth_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {"page": 10, "category": "地の文中心"},
                    {"page": 70, "category": "数式中心"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (truth_dir / "page_0010.md").write_text("無限に続く。", encoding="utf-8")
    (pred_dir / "page_0010.md").write_text("無制限に続く。", encoding="utf-8")
    (truth_dir / "page_0070.md").write_text("$$\\sqrt[4]{a}$$", encoding="utf-8")
    (pred_dir / "page_0070.md").write_text(
        "$$\\sqrt[3]{a}$$\n\n$$\\begin{array}{r}1\\\\ \\cline{1-1}\\end{array}$$",
        encoding="utf-8",
    )
    return truth_dir, pred_dir


class TestEvaluateDataset:
    """評価セット全体の評価。"""

    def test_reports_each_page_with_katex_errors(self, dataset: tuple[Path, Path]) -> None:
        """ページごとの指標と、評価対象出力の KaTeX エラーを記録する。"""
        truth_dir, pred_dir = dataset

        report = evaluate_dataset(truth_dir, pred_dir, katex_checker=_fake_katex)

        assert [row.page for row in report.pages] == [10, 70]
        assert report.pages[0].scores.cer.edits == 1
        assert report.pages[1].katex_errors == ("Undefined control sequence: \\cline",)

    def test_summarizes_by_category_and_total(self, dataset: tuple[Path, Path]) -> None:
        """層（カテゴリ）別と全体の集計を出す。"""
        truth_dir, pred_dir = dataset

        report = evaluate_dataset(truth_dir, pred_dir, katex_checker=_fake_katex)

        assert set(report.by_category) == {"地の文中心", "数式中心"}
        assert report.by_category["数式中心"].scores.math.truth_count == 1
        assert report.total.katex_error_count == 1
        assert report.total.scores.cer.truth_length == len("無限に続く。")

    def test_to_dict_contains_computed_metrics(self, dataset: tuple[Path, Path]) -> None:
        """JSON 出力用の辞書に CER・F1 などの計算済み指標を含める。"""
        truth_dir, pred_dir = dataset

        data = evaluate_dataset(truth_dir, pred_dir, katex_checker=_fake_katex).to_dict()

        assert data["total"]["cer"] == pytest.approx(1 / 6)
        assert data["total"]["math_f1"] == 0.0
        assert data["total"]["katex_errors"] == 1
        assert data["pages"][1]["katex_error_messages"] == ["Undefined control sequence: \\cline"]


class TestCli:
    """評価 CLI。"""

    def test_prints_summary_and_writes_json(
        self, dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """集計表を表示し、--json 指定時は結果を JSON ファイルに書き出す。"""
        truth_dir, pred_dir = dataset
        json_path = tmp_path / "result.json"
        monkeypatch.setattr(cli, "check_katex", _fake_katex)

        result = CliRunner().invoke(
            cli.app,
            ["--truth", str(truth_dir), "--pred", str(pred_dir), "--json", str(json_path)],
        )

        assert result.exit_code == 0, result.output
        assert "全体" in result.output
        assert json.loads(json_path.read_text(encoding="utf-8"))["total"]["katex_errors"] == 1

    def test_missing_prediction_fails(
        self, dataset: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """評価対象のページが欠けていれば異常終了する。"""
        truth_dir, pred_dir = dataset
        (pred_dir / "page_0070.md").unlink()
        monkeypatch.setattr(cli, "check_katex", _fake_katex)

        result = CliRunner().invoke(cli.app, ["--truth", str(truth_dir), "--pred", str(pred_dir)])

        assert result.exit_code != 0
        assert "page_0070" in result.output
