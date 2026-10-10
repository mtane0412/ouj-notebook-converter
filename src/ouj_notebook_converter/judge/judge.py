"""仕様: 誤読候補を原本画像と照合して LLM（Gemini）に判定させる。

- build_prompt: 候補の文字列・前後の文脈・raw.md の抜粋・検出器のヒントから判定用プロンプトを作る
- parse_reply: LLM の構造化出力（JSON）を検証して JudgmentReply にする
- GeminiLlmClient: Gemini API を呼ぶクライアント。一時的なエラー（429・5xx）は指数バックオフで再試行する
- judge_candidate / judge_all: 候補ごとに判定する。API 失敗・応答不正は握りつぶさず Judgment.error に
  記録し、verdict は uncertain にする（呼び出し側が件数を報告し、人の確認に回す）

LLM のクライアントは LlmClient プロトコルで差し替えられる（テストではモックする）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from ouj_notebook_converter.judge.models import (
    Confidence,
    JudgeCandidate,
    Judgment,
    Usage,
    Verdict,
)

ImageKind = Literal["crop", "page"]

MAX_ATTEMPTS = 4
RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_SECONDS = 2.0

_PROMPT = """\
あなたは日本語の数学教科書の OCR 校正者です。添付の画像は原本（スキャン）{image_description}です。
OCR（Gemini）の出力 Markdown と、検出器が「誤読かもしれない」と指摘した箇所を示します。
原本画像と OCR 出力を一字ずつ照合し、指摘箇所が原本と違っているか（誤読か）を判定してください。
判定は、画像から書き写した文字列（transcription）と OCR 出力を比べて行ってください。

## 判定の基準
- misread（誤読）: OCR の文字・数字・記号・添字・指数・括弧が原本と違う。脱落・余分な文字も含む
- correct（正しい）: OCR は原本どおりで、検出器の指摘は誤検出
- uncertain（判断不能）: 画像が不鮮明などで、原本の表記を確定できない
- 次は誤読ではない: LaTeX の書き方の違い（空白、\\left \\right、\\dfrac と \\frac など）、
  全角と半角の違い、原本自体の誤植をそのまま写している場合
- 検出器のヒントは手がかりにすぎません。別の OCR（yomitoku）の読みも誤ることが多いので、必ず画像で確かめてください
- 画像に見えない部分の内容を推測で補わないでください

## 指摘された箇所
- ページ: {page}
- 検出器: {source}（{kind}）
- ヒント: {hint}
{details}
## OCR 出力の抜粋{scope_note}
```
{excerpt}
```

## 回答（JSON。次の順に書く）
- transcription: まず、上の OCR 出力を見ずに、画像だけを見て、指摘箇所を含む 1〜2 文を一字ずつ書き写した文字列（数式は LaTeX）。\
OCR 出力の文が自然に見えても、画像の文字と一字ずつ照合する（OCR は「分」「に」などの 1 文字の脱落や置換を起こす）。\
画像に見えない部分は書き写さない
- verdict: misread / correct / uncertain
- confidence: high（画像で明確に確認できた）/ medium / low
- before: verdict が misread のとき、上の抜粋にそのまま現れる文字列（空白・改行・記号も一字一句同じ）。\
誤読箇所を含み、抜粋中で 1 か所に決まる最小限の長さにする（前後の数文字を含めてよい）。それ以外は空文字
- after: verdict が misread のとき、before のうち誤読箇所だけを原本どおりに直した文字列。\
LaTeX の書き方は OCR の出力にそろえる。それ以外は空文字
- reason: 原本画像で見えた内容を 1〜2 文で
"""


class JudgmentReply(BaseModel):
    """LLM の構造化出力のスキーマ（Gemini の response_schema にも使う）。"""

    transcription: str
    verdict: Verdict
    confidence: Confidence
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class LlmReply:
    """LLM の応答（JSON 文字列）とトークン使用量。"""

    text: str
    usage: Usage


class LlmClient(Protocol):
    """判定に使う LLM クライアントのインターフェース。"""

    model: str

    def generate(self, prompt: str, images: Sequence[bytes]) -> LlmReply:
        """プロンプトと JPEG 画像を送り、応答を返す。失敗時は RuntimeError。"""
        ...


def build_prompt(candidate: JudgeCandidate, *, image_kind: ImageKind) -> str:
    """判定用のプロンプトを作る。"""
    details: list[str] = []
    if candidate.text:
        details.append(f"- 疑わしい文字列: {candidate.text}")
    if candidate.context_before or candidate.context_after:
        details.append(
            f"- 前後の文脈（空白・記法を除いた地の文）: …{candidate.context_before}"
            f"【ここ】{candidate.context_after}…"
        )
    if candidate.yomitoku_text:
        details.append(f"- 別の OCR（yomitoku）が同じ箇所に読んだ文字列: {candidate.yomitoku_text}")
    return _PROMPT.format(
        image_description=(
            "ページのうち該当箇所の切り出し（前後の行を含む）"
            if image_kind == "crop"
            else "のページ全体"
        ),
        page=candidate.page,
        source=candidate.source,
        kind=candidate.kind,
        hint=candidate.hint,
        details="\n".join(details) + ("\n" if details else ""),
        scope_note=(
            "（該当行と前後の行）"
            if candidate.excerpt_scope == "lines"
            else "（該当行を特定できなかったためページ全体）"
        ),
        excerpt=candidate.excerpt,
    )


def parse_reply(text: str) -> JudgmentReply:
    """LLM の応答（JSON 文字列）を検証して解釈する。

    Raises:
        ValueError: JSON として読めない、またはスキーマに合わない場合。
    """
    try:
        return JudgmentReply.model_validate_json(text)
    except ValidationError as e:
        raise ValueError(f"LLM の応答を解釈できません: {e}") from e


class GeminiLlmClient:
    """Gemini API で判定するクライアント。構造化出力（JSON）を要求する。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        genai_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if genai_client is None:
            if not api_key:
                raise ValueError("Gemini の API キーが必要です（GEMINI_API_KEY）")
            from google import genai

            genai_client = genai.Client(api_key=api_key)
        self.model = model
        self._client = genai_client
        self._sleep = sleep

    def generate(self, prompt: str, images: Sequence[bytes]) -> LlmReply:
        """画像（JPEG）とプロンプトを送り、JSON 応答とトークン使用量を返す。

        Raises:
            RuntimeError: 再試行しても API 呼び出しが失敗した場合。
        """
        from google.genai import types

        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=JudgmentReply
        )
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as e:
                retryable = getattr(e, "code", None) in RETRYABLE_CODES
                if retryable and attempt < MAX_ATTEMPTS - 1:
                    self._sleep(BACKOFF_BASE_SECONDS * 2**attempt)
                    continue
                raise RuntimeError(f"Gemini API 呼び出しに失敗しました: {e}") from e
            meta = response.usage_metadata
            return LlmReply(
                text=response.text or "",
                usage=Usage(
                    input_tokens=meta.prompt_token_count or 0,
                    # 思考トークンは出力と同じ単価で課金される
                    output_tokens=(meta.candidates_token_count or 0)
                    + (meta.thoughts_token_count or 0),
                ),
            )
        raise AssertionError("到達しない")  # pragma: no cover


def judge_candidate(
    client: LlmClient,
    candidate: JudgeCandidate,
    images: Sequence[bytes],
    *,
    image_kind: ImageKind,
) -> Judgment:
    """1 候補を判定する。失敗時は verdict=uncertain で error に理由を記録する。"""
    try:
        reply = client.generate(build_prompt(candidate, image_kind=image_kind), images)
    except RuntimeError as e:
        return Judgment(
            candidate_id=candidate.id,
            model=client.model,
            verdict="uncertain",
            confidence="low",
            error=str(e),
        )
    try:
        parsed = parse_reply(reply.text)
    except ValueError as e:
        return Judgment(
            candidate_id=candidate.id,
            model=client.model,
            verdict="uncertain",
            confidence="low",
            usage=reply.usage,
            error=str(e),
        )
    return Judgment(
        candidate_id=candidate.id,
        model=client.model,
        verdict=parsed.verdict,
        confidence=parsed.confidence,
        before=parsed.before,
        after=parsed.after,
        reason=parsed.reason,
        transcription=parsed.transcription,
        usage=reply.usage,
    )


def judge_all(
    client: LlmClient,
    items: Sequence[tuple[JudgeCandidate, Sequence[bytes], ImageKind]],
    *,
    workers: int = 4,
) -> list[Judgment]:
    """全候補を並列に判定し、入力と同じ順序で結果を返す。"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(judge_candidate, client, candidate, images, image_kind=kind)
            for candidate, images, kind in items
        ]
        return [future.result() for future in futures]
