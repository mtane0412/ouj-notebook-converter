"""仕様: judge.judge（プロンプト作成・応答の解釈・Gemini クライアント）のユニットテスト。

API 呼び出しはすべてモックする。実際の Gemini は呼ばない。
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ouj_notebook_converter.judge.judge import (
    GeminiLlmClient,
    LlmReply,
    build_prompt,
    judge_all,
    judge_candidate,
    parse_reply,
)
from ouj_notebook_converter.judge.models import JudgeCandidate, Usage
from ouj_notebook_converter.judge.pricing import estimate_cost


def _candidate(**overrides: Any) -> JudgeCandidate:
    base: dict[str, Any] = {
        "id": "p0060-cross_ocr_diff-1",
        "page": 60,
        "source": "cross_ocr_diff",
        "kind": "gemini_only",
        "text": "制",
        "hint": "別の OCR（yomitoku）はこの箇所に文字を読んでいません",
        "excerpt": "数の並びが無制限に繰り返される",
        "excerpt_scope": "lines",
        "line_index": 0,
        "context_before": "かの数の並びが無",
        "context_after": "限に繰り返されて",
        "bbox": (110, 802, 996, 1433),
        "location": "yomitoku_paragraph",
    }
    return JudgeCandidate(**{**base, **overrides})


class FakeClient:
    """決まった応答を返す LLM クライアント。呼び出し内容を記録する。"""

    model = "fake-model"

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, images: list[bytes]) -> LlmReply:
        self.calls.append((prompt, len(images)))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LlmReply(text=reply, usage=Usage(input_tokens=1000, output_tokens=50))


_MISREAD_JSON = json.dumps(
    {
        "transcription": "数の並びが無限に繰り返される",
        "verdict": "misread",
        "confidence": "high",
        "before": "無制限に",
        "after": "無限に",
        "reason": "原本では「無限に」と印刷されている",
    },
    ensure_ascii=False,
)


def test_build_prompt_は候補の情報をすべて含む() -> None:
    prompt = build_prompt(_candidate(), image_kind="crop")
    assert "p.60" in prompt or "60" in prompt
    assert "数の並びが無制限に繰り返される" in prompt
    assert "別の OCR（yomitoku）はこの箇所に文字を読んでいません" in prompt
    assert "かの数の並びが無" in prompt
    assert "切り出し" in prompt


def test_build_prompt_は先に画像だけから書き写させる() -> None:
    """OCR の文を信じて「正しい」と答える偏りを減らすため、まず画像だけを書き写させる。"""
    prompt = build_prompt(_candidate(), image_kind="crop")
    assert "transcription" in prompt
    assert "書き写" in prompt


def test_build_prompt_は画像がページ全体なら_その旨を書く() -> None:
    assert "ページ全体" in build_prompt(_candidate(), image_kind="page")


def test_build_prompt_は_yomitoku_の読みを含める() -> None:
    prompt = build_prompt(_candidate(kind="replace", yomitoku_text="限"), image_kind="crop")
    assert "限" in prompt


def test_parse_reply_は構造化された応答を解釈する() -> None:
    reply = parse_reply(_MISREAD_JSON)
    assert reply.verdict == "misread"
    assert reply.confidence == "high"
    assert reply.before == "無制限に"
    assert reply.after == "無限に"
    assert reply.transcription == "数の並びが無限に繰り返される"


def test_parse_reply_は不正な応答でエラーにする() -> None:
    with pytest.raises(ValueError, match="応答"):
        parse_reply('{"verdict": "たぶん誤読"}')
    with pytest.raises(ValueError, match="応答"):
        parse_reply("JSON ではない")


def test_judge_candidate_は応答を_Judgment_にする() -> None:
    client = FakeClient([_MISREAD_JSON])
    judgment = judge_candidate(client, _candidate(), [b"jpeg"], image_kind="crop")
    assert judgment.candidate_id == "p0060-cross_ocr_diff-1"
    assert judgment.model == "fake-model"
    assert judgment.verdict == "misread"
    assert judgment.after == "無限に"
    assert judgment.transcription == "数の並びが無限に繰り返される"
    assert judgment.usage == Usage(input_tokens=1000, output_tokens=50)
    assert judgment.error == ""
    assert client.calls[0][1] == 1


def test_judge_candidate_は失敗を握りつぶさず_error_に記録する() -> None:
    client = FakeClient([RuntimeError("API が落ちています")])
    judgment = judge_candidate(client, _candidate(), [b"jpeg"], image_kind="crop")
    assert judgment.verdict == "uncertain"
    assert "API が落ちています" in judgment.error


def test_judge_candidate_は応答が不正でも_error_に記録する() -> None:
    client = FakeClient(["これは JSON ではありません"])
    judgment = judge_candidate(client, _candidate(), [b"jpeg"], image_kind="crop")
    assert judgment.verdict == "uncertain"
    assert "応答" in judgment.error


def test_judge_all_は候補の順に判定結果を返す() -> None:
    client = FakeClient([_MISREAD_JSON, _MISREAD_JSON])
    first, second = _candidate(id="a"), _candidate(id="b")
    results = judge_all(client, [(first, [b"x"], "crop"), (second, [b"y"], "page")], workers=1)
    assert [j.candidate_id for j in results] == ["a", "b"]


def test_GeminiLlmClient_は画像とプロンプトを送り_使用量を返す() -> None:
    sent: dict[str, Any] = {}

    class FakeModels:
        def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
            sent.update(model=model, contents=contents, config=config)
            return SimpleNamespace(
                text=_MISREAD_JSON,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1200, candidates_token_count=80, thoughts_token_count=300
                ),
            )

    client = GeminiLlmClient(
        model="gemini-3.8-flash", genai_client=SimpleNamespace(models=FakeModels())
    )
    reply = client.generate("プロンプト", [b"\xff\xd8jpeg"])
    assert reply.text == _MISREAD_JSON
    # 出力トークンには思考トークンを含める（課金上は出力と同じ単価のため）
    assert reply.usage == Usage(input_tokens=1200, output_tokens=380)
    assert sent["model"] == "gemini-3.8-flash"
    assert len(sent["contents"]) == 2  # 画像 1 枚 + プロンプト
    assert sent["config"].response_mime_type == "application/json"


def test_GeminiLlmClient_は一時的なエラーを再試行する() -> None:
    class TransientError(Exception):
        code = 503

    attempts: list[int] = []

    class FakeModels:
        def generate_content(self, **_: Any) -> Any:
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientError("overloaded")
            return SimpleNamespace(
                text=_MISREAD_JSON,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1, candidates_token_count=1, thoughts_token_count=None
                ),
            )

    sleeps: list[float] = []
    client = GeminiLlmClient(
        model="m", genai_client=SimpleNamespace(models=FakeModels()), sleep=sleeps.append
    )
    assert client.generate("p", []).text == _MISREAD_JSON
    assert len(attempts) == 3
    assert len(sleeps) == 2


def test_GeminiLlmClient_は再試行できないエラーをそのまま上げる() -> None:
    class FatalError(Exception):
        code = 400

    class FakeModels:
        def generate_content(self, **_: Any) -> Any:
            raise FatalError("不正なリクエスト")

    client = GeminiLlmClient(
        model="m", genai_client=SimpleNamespace(models=FakeModels()), sleep=lambda _: None
    )
    with pytest.raises(RuntimeError, match="不正なリクエスト"):
        client.generate("p", [])


def test_estimate_cost_は百万トークン単価で計算する() -> None:
    cost = estimate_cost("gemini-3.8-flash", Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(0.75 + 3.75)


def test_estimate_cost_は単価不明のモデルでNoneを返す() -> None:
    assert estimate_cost("unknown-model", Usage(1, 1)) is None
