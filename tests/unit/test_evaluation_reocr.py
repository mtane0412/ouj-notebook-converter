"""仕様: evaluation.reocr モジュール（評価セットのページだけを再 OCR する実験用ハーネス）のテスト。

Gemini API・PDF レンダリングはモックに差し替えるため、API キーも PDF も不要。
出力ディレクトリが評価コマンド（evaluation.dataset）でそのまま読めることも確認する。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from ouj_notebook_converter.evaluation.dataset import list_prediction_pages, read_prediction
from ouj_notebook_converter.evaluation.reocr import (
    PageRecord,
    app,
    resolve_output_dir,
    run_reocr,
)
from ouj_notebook_converter.plugins.ocr.gemini import GeminiAnalyzerResult, GeminiUsage


class _FakeApiError(Exception):
    """google.genai.errors.APIError 相当（code 属性を持つ）。"""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class _FakeAnalyzer:
    """ページ番号に応じた Markdown を返し、設定した失敗を順に発生させる Gemini 代役。"""

    def __init__(self, failures: list[Exception] | None = None, omit_usage: bool = False) -> None:
        self.omit_usage = omit_usage
        self.failures = list(failures or [])
        self.last_usage: GeminiUsage | None = None
        self.call_count = 0

    def __call__(self, image: np.ndarray) -> tuple[GeminiAnalyzerResult, None, None]:
        self.call_count += 1
        if self.failures:
            error = self.failures.pop(0)
            raise RuntimeError("Gemini API 呼び出しに失敗しました") from error
        page = int(image[0, 0, 0])  # render_page がページ番号を画素値に埋め込む
        self.last_usage = (
            None
            if self.omit_usage
            else GeminiUsage(
                prompt_tokens=1000 + page,
                output_tokens=500,
                thinking_tokens=200,
                total_tokens=1700 + page,
            )
        )
        return GeminiAnalyzerResult(f"# {page} ページの再OCR結果"), None, None


def _render(page: int) -> np.ndarray:
    return np.full((2, 2, 3), page, dtype=np.uint8)


class TestRunReocr:
    def test_ページごとにraw_mdとusage_jsonを書き出す(self, tmp_path: Path) -> None:
        records = run_reocr(
            pages=[5, 70],
            render_page=_render,
            analyzer=_FakeAnalyzer(),
            out_dir=tmp_path,
            sleep=lambda _s: None,
        )

        assert [r.page for r in records] == [5, 70]
        assert (tmp_path / "page_0070" / "raw.md").read_text(encoding="utf-8") == (
            "# 70 ページの再OCR結果"
        )
        usage = json.loads((tmp_path / "page_0070" / "usage.json").read_text(encoding="utf-8"))
        assert usage["page"] == 70
        assert usage["prompt_tokens"] == 1070
        assert usage["output_tokens"] == 500
        assert usage["thinking_tokens"] == 200
        assert usage["attempts"] == 1
        assert usage["seconds"] >= 0

    def test_出力ディレクトリは評価コマンドがそのまま読める(self, tmp_path: Path) -> None:
        run_reocr(
            pages=[5, 70],
            render_page=_render,
            analyzer=_FakeAnalyzer(),
            out_dir=tmp_path,
            sleep=lambda _s: None,
        )

        assert list_prediction_pages(tmp_path) == [5, 70]
        assert "70 ページ" in read_prediction(tmp_path, 70)

    def test_処理時間は成功した呼び出しの経過時間を記録する(self, tmp_path: Path) -> None:
        ticks = iter([10.0, 12.5])  # 開始・終了

        records = run_reocr(
            pages=[5],
            render_page=_render,
            analyzer=_FakeAnalyzer(),
            out_dir=tmp_path,
            sleep=lambda _s: None,
            clock=lambda: next(ticks),
        )

        assert records[0].seconds == pytest.approx(2.5)

    def test_レート制限では待って再試行し試行回数を記録する(self, tmp_path: Path) -> None:
        waits: list[float] = []
        analyzer = _FakeAnalyzer(failures=[_FakeApiError(429), _FakeApiError(503)])

        records = run_reocr(
            pages=[5],
            render_page=_render,
            analyzer=analyzer,
            out_dir=tmp_path,
            sleep=waits.append,
            retry_wait_seconds=10.0,
        )

        assert records[0].attempts == 3
        assert analyzer.call_count == 3
        assert waits == [10.0, 20.0]  # 指数的に待ち時間が伸びる

    def test_再試行回数を超えたら例外を送出する(self, tmp_path: Path) -> None:
        analyzer = _FakeAnalyzer(failures=[_FakeApiError(429)] * 3)

        with pytest.raises(RuntimeError, match="再試行"):
            run_reocr(
                pages=[5],
                render_page=_render,
                analyzer=analyzer,
                out_dir=tmp_path,
                sleep=lambda _s: None,
                max_retries=2,
            )

    def test_再試行対象外のエラーは待たずに即座に失敗する(self, tmp_path: Path) -> None:
        analyzer = _FakeAnalyzer(failures=[_FakeApiError(400)])

        with pytest.raises(RuntimeError, match="Gemini API 呼び出しに失敗"):
            run_reocr(
                pages=[5],
                render_page=_render,
                analyzer=analyzer,
                out_dir=tmp_path,
                sleep=lambda _s: pytest.fail("待機してはいけない"),
            )
        assert analyzer.call_count == 1

    def test_完了済みページはAPIを呼ばずusage_jsonから記録を復元する(self, tmp_path: Path) -> None:
        run_reocr(
            pages=[5],
            render_page=_render,
            analyzer=_FakeAnalyzer(),
            out_dir=tmp_path,
            sleep=lambda _s: None,
        )
        analyzer = _FakeAnalyzer()

        records = run_reocr(
            pages=[5, 70],
            render_page=_render,
            analyzer=analyzer,
            out_dir=tmp_path,
            sleep=lambda _s: None,
        )

        assert analyzer.call_count == 1  # 70 ページだけ
        assert [r.page for r in records] == [5, 70]
        assert records[0].usage is not None
        assert records[0].usage.prompt_tokens == 1005

    def test_usage_metadataが取れないと失敗する(self, tmp_path: Path) -> None:
        analyzer = _FakeAnalyzer(omit_usage=True)

        with pytest.raises(RuntimeError, match="usage_metadata"):
            run_reocr(
                pages=[5],
                render_page=_render,
                analyzer=analyzer,
                out_dir=tmp_path,
                sleep=lambda _s: None,
            )


class TestResolveOutputDir:
    def test_run_id未指定ならoutそのもの(self, tmp_path: Path) -> None:
        assert resolve_output_dir(tmp_path, None) == tmp_path

    def test_run_id指定ならout配下のサブディレクトリ(self, tmp_path: Path) -> None:
        assert resolve_output_dir(tmp_path, "run1") == tmp_path / "run1"

    def test_パス区切りを含むrun_idは拒否する(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="run-id"):
            resolve_output_dir(tmp_path, "../x")


class TestCli:
    def _truth(self, tmp_path: Path) -> Path:
        truth = tmp_path / "eval"
        truth.mkdir()
        (truth / "manifest.json").write_text(
            json.dumps(
                {
                    "pages": [
                        {"page": 5, "category": "地の文中心"},
                        {"page": 70, "category": "数式中心"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return truth

    def test_評価セットのページだけを再OCRしrun_jsonを出力する(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        analyzer = _FakeAnalyzer()
        factory = mocker.patch(
            "ouj_notebook_converter.evaluation.reocr.create_gemini_analyzer",
            return_value=analyzer,
        )
        mocker.patch(
            "ouj_notebook_converter.evaluation.reocr.render_pdf_page",
            side_effect=lambda _pdf, page, _dpi: _render(page),
        )
        mocker.patch("ouj_notebook_converter.evaluation.reocr.time.sleep")
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("実験用のプロンプト", encoding="utf-8")
        pdf = tmp_path / "原本.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out = tmp_path / "out"

        result = CliRunner().invoke(
            app,
            [
                "--truth", str(self._truth(tmp_path)),
                "--pdf", str(pdf),
                "--dpi", "300",
                "--model", "gemini-3.1-pro-preview",
                "--prompt-file", str(prompt_file),
                "--run-id", "run2",
                "--out", str(out),
                "--api-key", "テスト用APIキー",
            ],
        )  # fmt: skip

        assert result.exit_code == 0, result.output
        factory.assert_called_once_with(
            api_key="テスト用APIキー", model="gemini-3.1-pro-preview", prompt="実験用のプロンプト"
        )
        assert list_prediction_pages(out / "run2") == [5, 70]
        summary = json.loads((out / "run2" / "run.json").read_text(encoding="utf-8"))
        assert summary["dpi"] == 300
        assert summary["model"] == "gemini-3.1-pro-preview"
        assert summary["run_id"] == "run2"
        assert summary["prompt_file"] == str(prompt_file)
        assert summary["totals"]["pages"] == 2
        assert summary["totals"]["prompt_tokens"] == 1005 + 1070

    def test_APIキーが無ければエラー終了する(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        pdf = tmp_path / "原本.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        result = CliRunner().invoke(
            app,
            [
                "--truth",
                str(self._truth(tmp_path)),
                "--pdf",
                str(pdf),
                "--out",
                str(tmp_path / "o"),
            ],
        )

        assert result.exit_code == 1


def test_PageRecordは不変(tmp_path: Path) -> None:
    record = PageRecord(page=1, seconds=1.0, attempts=1, usage=None)
    with pytest.raises(AttributeError):
        record.page = 2  # type: ignore[misc]
