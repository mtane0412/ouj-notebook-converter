"""仕様: `python -m ouj_notebook_converter.judge` の CLI のユニットテスト。

collect → judge → export の流れを、Gemini API をモックして通しで確かめる。
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from ouj_notebook_converter.corrections import load_corrections
from ouj_notebook_converter.judge import cli
from ouj_notebook_converter.judge.judge import LlmReply
from ouj_notebook_converter.judge.models import Usage

_RAW = "数の並びが無制限に繰り返される。\n"
_REPLY = json.dumps(
    {
        "transcription": "数の並びが無限に繰り返される",
        "verdict": "misread",
        "confidence": "high",
        "before": "無制限に",
        "after": "無限に",
        "reason": "原本では「無限に」",
    },
    ensure_ascii=False,
)


class FakeGeminiClient:
    """Gemini を呼ばず、決まった応答を返す。"""

    instances: ClassVar[list["FakeGeminiClient"]] = []

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key
        self.prompts: list[str] = []
        FakeGeminiClient.instances.append(self)

    def generate(self, prompt: str, images: list[bytes]) -> LlmReply:
        self.prompts.append(prompt)
        return LlmReply(text=_REPLY, usage=Usage(input_tokens=2000, output_tokens=100))


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """p.1 に「無制限に」の誤読がある Gemini キャッシュ、yomitoku キャッシュ、原本 PDF を作る。"""
    from reportlab.pdfgen import canvas

    gemini = tmp_path / "gemini"
    (gemini / "page_0001").mkdir(parents=True)
    (gemini / "page_0001" / "raw.md").write_text(_RAW, encoding="utf-8")
    yomitoku = tmp_path / "yomitoku"
    (yomitoku / "page_0001").mkdir(parents=True)
    paragraph = {
        "box": [100, 300, 900, 340],
        "contents": "数の並びが無限に繰り返される。",
        "role": None,
    }
    (yomitoku / "page_0001" / "analysis.json").write_text(
        json.dumps(
            {"paragraphs": [paragraph], "figures": [], "tables": [], "words": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pdf = tmp_path / "book.pdf"
    pdf_canvas = canvas.Canvas(str(pdf), pagesize=(595, 842))
    pdf_canvas.drawString(50, 400, "test")
    pdf_canvas.save()
    return {"gemini": gemini, "yomitoku": yomitoku, "pdf": pdf, "tmp": tmp_path}


def _invoke(*args: str | Path) -> Any:
    return CliRunner().invoke(cli.app, [str(a) for a in args])


def test_collect_judge_export_を通しで実行できる(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiLlmClient", FakeGeminiClient)
    tmp = workspace["tmp"]

    result = _invoke(
        "collect",
        "--pred", workspace["gemini"],
        "--yomitoku", workspace["yomitoku"],
        "--source", "cross_ocr_diff",
        "--out", tmp / "candidates.json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    candidates = json.loads((tmp / "candidates.json").read_text(encoding="utf-8"))
    assert candidates["candidate_count"] == 1

    result = _invoke(
        "judge",
        "--candidates", tmp / "candidates.json",
        "--pdf", workspace["pdf"],
        "--model", "gemini-3.8-flash",
        "--crops-dir", tmp / "crops",
        "--out", tmp / "judgments.json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    judgments = json.loads((tmp / "judgments.json").read_text(encoding="utf-8"))
    assert judgments["summary"]["input_tokens"] == 2000
    assert judgments["summary"]["cost_usd"] == pytest.approx((2000 * 0.75 + 100 * 3.75) / 1e6)
    assert FakeGeminiClient.instances[-1].api_key == "test-key"
    assert any((tmp / "crops").glob("*.jpg"))

    result = _invoke(
        "export",
        "--candidates", tmp / "candidates.json",
        "--judgments", tmp / "judgments.json",
        "--pred", workspace["gemini"],
        "--out-dir", tmp / "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    corrections = load_corrections(tmp / "out" / "corrections.json")
    assert corrections[1][0].after == "無限に"


def test_export_は複数の判定が一致したものだけを自動適用する(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    tmp = workspace["tmp"]
    _invoke(
        "collect",
        "--pred", workspace["gemini"],
        "--yomitoku", workspace["yomitoku"],
        "--source", "cross_ocr_diff",
        "--out", tmp / "candidates.json",
    )  # fmt: skip

    class DisagreeingClient(FakeGeminiClient):
        """切り出し画像では誤読、ページ全体では原本どおりと答えるクライアント。"""

        def generate(self, prompt: str, images: list[bytes]) -> LlmReply:
            reply = json.dumps(
                {
                    "transcription": "数の並びが無制限に繰り返される",
                    "verdict": "correct",
                    "confidence": "high",
                    "before": "",
                    "after": "",
                    "reason": "原本どおり",
                },
                ensure_ascii=False,
            )
            return LlmReply(text=reply, usage=Usage(input_tokens=1000, output_tokens=50))

    for client, mode in [(FakeGeminiClient, "crop"), (DisagreeingClient, "page")]:
        monkeypatch.setattr(cli, "GeminiLlmClient", client)
        result = _invoke(
            "judge",
            "--candidates", tmp / "candidates.json",
            "--pdf", workspace["pdf"],
            "--image-mode", mode,
            "--out", tmp / f"judgments_{mode}.json",
        )  # fmt: skip
        assert result.exit_code == 0, result.output

    result = _invoke(
        "export",
        "--candidates", tmp / "candidates.json",
        "--judgments", tmp / "judgments_crop.json",
        "--judgments", tmp / "judgments_page.json",
        "--pred", workspace["gemini"],
        "--out-dir", tmp / "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert load_corrections(tmp / "out" / "corrections.json") == {}
    decisions = json.loads((tmp / "out" / "decisions.json").read_text(encoding="utf-8"))
    assert decisions[0]["decision"] == "review"


def test_evaluate_は判定の正誤と修正適用後の指標を出す(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiLlmClient", FakeGeminiClient)
    # KaTeX（Node.js）は使わず、数式のエラーは無いものとして扱う
    monkeypatch.setattr(cli, "check_katex", lambda formulas: [None for _ in formulas])
    tmp = workspace["tmp"]
    truth = tmp / "truth"
    truth.mkdir()
    (truth / "manifest.json").write_text(
        json.dumps({"pages": [{"page": 1, "category": "地の文中心"}]}), encoding="utf-8"
    )
    (truth / "page_0001.md").write_text("数の並びが無限に繰り返される。\n", encoding="utf-8")
    known = tmp / "known.json"
    known.write_text(
        json.dumps(
            {
                "version": 1,
                "corrections": [{"page": 1, "before": "無制限に", "after": "無限に"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _invoke(
        "collect",
        "--pred", workspace["gemini"],
        "--yomitoku", workspace["yomitoku"],
        "--source", "cross_ocr_diff",
        "--out", tmp / "candidates.json",
    )  # fmt: skip
    _invoke(
        "judge",
        "--candidates", tmp / "candidates.json",
        "--pdf", workspace["pdf"],
        "--out", tmp / "judgments.json",
    )  # fmt: skip

    result = _invoke(
        "evaluate",
        "--truth", truth,
        "--pred", workspace["gemini"],
        "--candidates", tmp / "candidates.json",
        "--judgments", tmp / "judgments.json",
        "--known-corrections", known,
        "--out", tmp / "evaluation.json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads((tmp / "evaluation.json").read_text(encoding="utf-8"))
    model = next(iter(data["models"].values()))
    assert model["accuracy_eval"]["true_positive"] == 1
    assert model["accuracy_eval"]["fix_correct"] == 1
    assert model["wrong_auto_eval"] == []
    key = next(iter(data["models"]))
    assert data["scores"][f"{key}:auto"]["prose_cer"] < data["scores"]["baseline"]["prose_cer"]


def test_judge_は_API_キーが無ければ失敗する(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (workspace["tmp"] / "candidates.json").write_text(
        json.dumps({"format_version": 1, "candidates": []}), encoding="utf-8"
    )
    result = _invoke(
        "judge",
        "--candidates", workspace["tmp"] / "candidates.json",
        "--pdf", workspace["pdf"],
        "--out", workspace["tmp"] / "judgments.json",
    )  # fmt: skip
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


def test_collect_は_yomitoku_が無いページがあれば失敗する(workspace: dict[str, Path]) -> None:
    (workspace["yomitoku"] / "page_0001" / "analysis.json").unlink()
    result = _invoke(
        "collect",
        "--pred", workspace["gemini"],
        "--yomitoku", workspace["yomitoku"],
        "--source", "cross_ocr_diff",
        "--out", workspace["tmp"] / "candidates.json",
    )  # fmt: skip
    assert result.exit_code == 1
    assert "analysis.json" in result.output
