"""仕様: 誤読候補の LLM 判定（issue #24）で使うデータ型と、候補ファイル（JSON）の読み書き。

- JudgeCandidate: 各検出器の誤読候補を正規化した共通形式。判定の入力になる
- Judgment: LLM の判定結果（誤読か・正しい表記・確信度）と、トークン使用量
- 候補ファイルは `format_version` を持ち、形式が変わったときに後段が検知できるようにする
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CANDIDATES_FORMAT_VERSION = 1

# 候補を出した検出器。連番検査（numbering）は欠落の報告で、置換の対象となる文字列が無いため含めない
Source = Literal["equation_check", "rare_word", "cross_ocr_diff", "katex_error"]
ALL_SOURCES: tuple[Source, ...] = ("equation_check", "rare_word", "cross_ocr_diff", "katex_error")

BBox = tuple[int, int, int, int]

# 画像上の位置の特定方法
#   yomitoku_word / yomitoku_paragraph: 検出器が yomitoku の座標を持っていた
#   line_alignment: Gemini の行を yomitoku の段落に対応付けた
#   band_between: 数式だけの行を、前後の対応付けできた段落に挟まれた帯として推定した
#   page: 特定できず、ページ全体を渡す
LocationMethod = Literal[
    "yomitoku_word", "yomitoku_paragraph", "line_alignment", "band_between", "page"
]

Verdict = Literal["misread", "correct", "uncertain"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class JudgeCandidate:
    """LLM に判定させる誤読候補 1 件。

    Attributes:
        id: 候補の識別子（`p0070-equation_check-1` の形式）。
        page: PDF のページ番号（1 始まり）。
        source: 候補を出した検出器。
        kind: 検出器ごとの種別（replace / gemini_only / rare_word / violated など）。
        text: Gemini の出力中の疑わしい文字列（yomitoku_only など空のこともある）。
        hint: 検出器が見つけた根拠（人が読む説明文）。
        excerpt: raw.md の該当行と前後 1 行。LLM はこの中から置換前の文字列を選ぶ。
        excerpt_scope: `lines` なら該当行の抜粋、`page` なら該当行を特定できずページ全体。
        line_index: 該当行の 0 始まりの行番号（特定できなければ None）。
        context_before / context_after: 疑わしい文字列の前後の文脈（地の文の正規化形）。
        yomitoku_text: 別 OCR（yomitoku）が同じ箇所に読んだ文字列。
        bbox: 原本画像（200 DPI）上の範囲 (x0, y0, x1, y1)。決まらなければ None。
        location: bbox の特定方法。bbox が無ければ `page`。
    """

    id: str
    page: int
    source: Source
    kind: str
    text: str
    hint: str
    excerpt: str
    excerpt_scope: Literal["lines", "page"]
    line_index: int | None
    context_before: str = ""
    context_after: str = ""
    yomitoku_text: str = ""
    bbox: BBox | None = None
    location: LocationMethod = "page"

    def to_dict(self) -> dict[str, Any]:
        """JSON に書き出すための辞書を返す。"""
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeCandidate:
        """to_dict の逆変換。"""
        bbox = data.get("bbox")
        return cls(**{**data, "bbox": tuple(bbox) if bbox is not None else None})


@dataclass(frozen=True)
class Usage:
    """1 回の API 呼び出しのトークン使用量。output_tokens には思考トークンを含める。"""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class Judgment:
    """1 候補に対する LLM の判定結果。

    Attributes:
        candidate_id: 判定した候補の id。
        model: 判定に使ったモデル名。
        verdict: misread（誤読）／correct（原本どおりで誤読ではない）／uncertain（判断できない）。
        confidence: 判定の確信度。
        before: 誤読のとき、抜粋中の置換前の文字列（そのままの部分文字列）。
        after: 誤読のとき、原本どおりの置換後の文字列。
        reason: 判定の理由（原本画像で見えた内容）。
        transcription: LLM が OCR 出力を見ずに原本画像から書き写した文字列（確認の手がかり）。
        usage: トークン使用量。
        error: API 呼び出しや応答の解釈に失敗したときのメッセージ（失敗時は verdict が uncertain）。
    """

    candidate_id: str
    model: str
    verdict: Verdict
    confidence: Confidence
    before: str = ""
    after: str = ""
    reason: str = ""
    transcription: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON に書き出すための辞書を返す。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Judgment:
        """to_dict の逆変換。"""
        return cls(**{**data, "usage": Usage(**data["usage"])})


def save_candidates(path: Path, candidates: list[JudgeCandidate]) -> None:
    """候補をファイル（format_version 付き JSON）に書き出す。"""
    data = {
        "format_version": CANDIDATES_FORMAT_VERSION,
        "candidate_count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidates(path: Path) -> list[JudgeCandidate]:
    """候補ファイルを読み込む。

    Raises:
        FileNotFoundError: ファイルが無い場合。
        ValueError: JSON として読めない、または format_version が違う場合。
    """
    if not path.is_file():
        raise FileNotFoundError(f"候補ファイルが見つかりません: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"候補ファイルを JSON として読めません: {path}: {e}") from e
    if not isinstance(data, dict) or data.get("format_version") != CANDIDATES_FORMAT_VERSION:
        raise ValueError(
            f"候補ファイルの format_version は {CANDIDATES_FORMAT_VERSION} である必要があります: {path}"
        )
    return [JudgeCandidate.from_dict(item) for item in data["candidates"]]
