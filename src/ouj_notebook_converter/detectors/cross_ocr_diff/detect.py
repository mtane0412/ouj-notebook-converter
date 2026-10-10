"""仕様: Gemini と yomitoku の OCR 結果を日本語部分だけで比較し、食い違いを誤読候補として抽出する（issue #23）。

処理の流れ（1 ページ分）:
  1. Gemini の Markdown から地の文を取り出し、日本語（かな・漢字・長音符）だけの比較用文字列を作る。
     数式・数字・英字・記号は区切り文字 1 つに置き換える（yomitoku は数式の構造を読めないため比較しない）
  2. yomitoku の analysis.json の段落ごとに、同じ方法で日本語の連続（run）に分ける
  3. run が Gemini の比較用文字列にそのまま含まれていれば一致。含まれていなければ、最長一致を足がかりに
     Gemini 側の対応箇所を切り出して文字単位で比較し、食い違いを誤読候補にする

ノイズ除去:
  - 図の領域にある段落（図中文字）と、ページ上部の帯にある段落（柱）は比較しない
  - 句読点・括弧は比較から外す（yomitoku は「、」と「,」などを揺らす）
  - 「一」（漢数字）と「ー」（長音符）、小書きの仮名と通常の仮名は同じ文字とみなす。
    yomitoku が「ー」だけを読んだ食い違いは候補にしない（数式のマイナスや分数線を「一」と読むため）
  - 数式・変数のあった位置（Gemini の数式の位置）に、yomitoku が 1〜2 文字の仮名・記号を読んだ食い違い
    と、段落に取り込まれた図番号の「図」は候補にしない
  - 置き換えの片側が 5 文字を超える食い違いは、読む順序の違いによる対応の取り違えとして候補にしない
  - 2 文字以下の run は偶然の一致・不一致が多いため比較しない

注意事項:
  - 候補は「どちらかが誤読している箇所」であり、どちらが正しいかは判定しない（後段の LLM 判定に委ねる）
  - 「一」を落とした Gemini の誤読や、小書きの仮名の誤読は、上記の規則のため検出できない
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ouj_notebook_converter.evaluation.dataset import list_prediction_pages, read_prediction
from ouj_notebook_converter.evaluation.markdown_parts import split_markdown
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN, split_code_fences

# 比較用文字列で日本語以外の連続を表す区切り文字（run の境界）
SEPARATOR = "\0"
# Gemini の地の文で数式があった位置を表す印（私用領域の文字。出力時は MATH_LABEL に置き換える）
MATH_MARKER = "\ue000"
MATH_LABEL = "〔数式〕"
# 比較から外す句読点・括弧（NFKC 正規化後の文字）
_DROPPED_CHARS = frozenset("、。，．,.・･·「」『』()[]【】〈〉《》“”\"'!?！？:：;；〜~…")
# 日本語の文字: ひらがな・カタカナ・漢字・々・〆・長音符
_JAPANESE_CHAR = re.compile(r"[ぁ-ゟァ-ヿ㐀-鿿々〆ー]")
# yomitoku が取り違えやすい「一」「ー」を揃えるための対応
# 小書きの仮名も通常の大きさに揃える（yomitoku は「よ」と「ょ」などを取り違える）
_VARIANT_FOLD = str.maketrans(
    {"一": "ー", **dict(zip("ぁぃぅぇぉゃゅょっゎ", "あいうえおやゆよつわ", strict=True))}
    | dict(zip("ァィゥェォャュョッヮ", "アイウエオヤユヨツワ", strict=True))
)
_DASH_CHARS = frozenset("ーｰ")

# 柱（ページ上部の章名・ページ番号）とみなす帯。段落の下端がこの y 座標（px）以下なら柱とする
# 根拠: 評価対象の書籍では柱の段落が y=60〜97 に現れ、本文の最上段は y=140 以降
HEADER_BAND_MAX_Y = 110
# 比較する run の最小文字数（これより短い run は偶然の一致・不一致が多い）
DEFAULT_MIN_RUN_LENGTH = 3
# 最長一致が run の何割以上なら Gemini 側の対応箇所とみなすか
ANCHOR_MIN_RATIO = 0.4
# 最長一致から Gemini 側の切り出し範囲を広げる余白（文字数）
WINDOW_PADDING = 2
# 置き換えの片側がこの文字数を超えたら、対応の取り違えとみなして候補にしない
MAX_REPLACE_LENGTH = 4
# 数式や変数の位置に yomitoku が読んだ仮名・記号とみなす、yomitoku 側の最大文字数
MAX_SYMBOL_NOISE_LENGTH = 2
# 図番号・表番号の見出し語（yomitoku は段落に取り込むことがある）
_CAPTION_LABELS = frozenset("図表")
# 誤読候補に付ける前後の文脈の文字数
DEFAULT_CONTEXT_CHARS = 15

CandidateKind = Literal["replace", "gemini_only", "yomitoku_only", "unmatched_run"]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DiffOptions:
    """比較の設定。ノイズ除去の効果を測るため、個別に無効化できる。

    Attributes:
        min_run_length: 比較する日本語の連続の最小文字数。
        exclude_figures: 図の領域にある段落を比較しない。
        exclude_header: ページ上部の帯（柱）にある段落を比較しない。
        normalize_variants: 「一」と「ー」、小書きの仮名と通常の仮名を同一視する。さらに、yomitoku 側が
            「ー」だけの食い違い（数式のマイナスや分数線の読み違い）は候補にしない。
        report_unmatched: Gemini に対応箇所が見つからない連続を unmatched_run として候補にする
            （既定では図中の記号などの誤検出が多いため、件数だけ数える）。
        context_chars: 候補に付ける前後の文脈の文字数。
    """

    min_run_length: int = DEFAULT_MIN_RUN_LENGTH
    exclude_figures: bool = True
    exclude_header: bool = True
    normalize_variants: bool = True
    report_unmatched: bool = False
    context_chars: int = DEFAULT_CONTEXT_CHARS


@dataclass(frozen=True)
class PageStats:
    """1 ページ分の処理件数（ノイズ除去の効果を測るための内訳）。"""

    paragraphs: int = 0
    paragraphs_in_figure: int = 0
    paragraphs_in_header: int = 0
    runs: int = 0
    runs_short: int = 0
    runs_exact: int = 0
    runs_different: int = 0
    runs_unmatched: int = 0
    ops_misaligned: int = 0

    def __add__(self, other: PageStats) -> PageStats:
        return PageStats(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class Candidate:
    """Gemini と yomitoku の食い違い 1 か所（誤読候補）。

    Attributes:
        page: ページ番号（1 始まり）。
        kind: replace（両方に文字があり異なる）／gemini_only（Gemini にだけ文字がある）／
            yomitoku_only（yomitoku にだけ文字がある）／unmatched_run（Gemini に対応する文章が見つからない）。
        gemini_text: Gemini 側の食い違い部分（yomitoku_only と unmatched_run では空）。
        gemini_before: Gemini 側の食い違いの直前の文脈（unmatched_run では空）。
        gemini_after: Gemini 側の食い違いの直後の文脈（unmatched_run では空）。
        gemini_span: Gemini の地の文（split_markdown().prose）での範囲。空の範囲は挿入位置を表す。
            unmatched_run では None。
        yomitoku_text: yomitoku 側の食い違い部分（gemini_only では空）。
        yomitoku_before: yomitoku 側の直前の文脈（同じ段落内）。
        yomitoku_after: yomitoku 側の直後の文脈（同じ段落内）。
        bbox: yomitoku 側の位置（x0, y0, x1, y1）。
        bbox_source: bbox が単語（word）か段落（paragraph）のどちらの座標か。
    """

    page: int
    kind: CandidateKind
    gemini_text: str
    gemini_before: str
    gemini_after: str
    gemini_span: tuple[int, int] | None
    yomitoku_text: str
    yomitoku_before: str
    yomitoku_after: str
    bbox: BBox
    bbox_source: Literal["word", "paragraph"]

    def to_dict(self) -> dict[str, Any]:
        """後段の LLM 判定への入力として JSON に書き出す辞書を返す。"""
        return {
            "page": self.page,
            "kind": self.kind,
            "gemini": {
                "text": self.gemini_text,
                "context_before": self.gemini_before,
                "context_after": self.gemini_after,
            },
            "yomitoku": {
                "text": self.yomitoku_text,
                "context_before": self.yomitoku_before,
                "context_after": self.yomitoku_after,
                "bbox": list(self.bbox),
                "bbox_source": self.bbox_source,
            },
        }


@dataclass(frozen=True)
class PageResult:
    """1 ページ分の比較結果。"""

    page: int
    candidates: tuple[Candidate, ...]
    stats: PageStats = field(default_factory=PageStats)


class _Paragraph(BaseModel):
    box: tuple[int, int, int, int]
    contents: str


class _Figure(BaseModel):
    box: tuple[int, int, int, int]


class _Word(BaseModel):
    content: str
    points: list[tuple[int, int]]


class _Analysis(BaseModel):
    paragraphs: list[_Paragraph]
    figures: list[_Figure] = []
    words: list[_Word] = []


def parse_analysis(raw: dict[str, Any], *, source: str) -> _Analysis:
    """yomitoku の analysis.json（読み込み済みの辞書）を検証して返す。

    Raises:
        ValueError: 段落の位置・内容など、必要な項目が無い場合。
    """
    try:
        return _Analysis.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"{source} の形式が不正です（analysis.json）: {e}") from e


def build_comparable(text: str, *, fold_variants: bool = True) -> tuple[str, list[int]]:
    """日本語部分だけの比較用文字列と、各文字の元の文字列での位置を作る。

    日本語以外の連続は SEPARATOR 1 文字に、句読点・括弧は取り除く。

    Args:
        text: NFKC 正規化済みで、空白を含まない文字列。
        fold_variants: 「一」を「ー」に、小書きの仮名を通常の仮名に揃えるか。

    Returns:
        (比較用文字列, 比較用文字列の各文字に対応する元の文字列の位置)。
    """
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in _DROPPED_CHARS:
            continue
        if _JAPANESE_CHAR.fullmatch(ch):
            chars.append(ch.translate(_VARIANT_FOLD) if fold_variants else ch)
            index_map.append(i)
        elif not chars or chars[-1] != SEPARATOR:
            chars.append(SEPARATOR)
            index_map.append(i)
    return "".join(chars), index_map


def compare_page(
    page: int,
    gemini_markdown: str,
    analysis: dict[str, Any] | _Analysis,
    options: DiffOptions | None = None,
) -> PageResult:
    """1 ページ分の Gemini と yomitoku を比較して誤読候補を返す。

    Args:
        page: ページ番号（1 始まり）。
        gemini_markdown: Gemini の Markdown（評価・検出器と同じく後処理済みのもの）。
        analysis: yomitoku の analysis.json（辞書、または parse_analysis の結果）。
        options: 比較の設定。

    Raises:
        ValueError: analysis の形式が不正な場合。
    """
    options = options or DiffOptions()
    parsed = (
        analysis
        if isinstance(analysis, _Analysis)
        else parse_analysis(analysis, source=f"ページ {page}")
    )
    prose = extract_gemini_prose(gemini_markdown)
    gemini_text, gemini_map = build_comparable(prose, fold_variants=options.normalize_variants)

    candidates: list[Candidate] = []
    stats = PageStats()
    for paragraph in parsed.paragraphs:
        stats = stats + PageStats(paragraphs=1)
        if options.exclude_figures and any(
            _center_in(paragraph.box, f.box) for f in parsed.figures
        ):
            stats = stats + PageStats(paragraphs_in_figure=1)
            continue
        if options.exclude_header and paragraph.box[3] <= HEADER_BAND_MAX_Y:
            stats = stats + PageStats(paragraphs_in_header=1)
            continue
        paragraph_result = _compare_paragraph(
            page, paragraph, parsed.words, prose, gemini_text, gemini_map, options
        )
        candidates.extend(paragraph_result[0])
        stats = stats + paragraph_result[1]
    return PageResult(page=page, candidates=tuple(candidates), stats=stats)


def extract_gemini_prose(markdown: str) -> str:
    """Gemini の Markdown から地の文を取り出す。数式のあった位置には MATH_MARKER を残す。

    評価の地の文（split_markdown().prose）と同じ正規化をするが、数式を完全には消さず、
    位置だけを印として残す（数式の位置に yomitoku が仮名や記号を読んだ食い違いを見分けるため）。
    数式内の \\text{...} の中身は印の直後に残る。コードフェンス内は変更しない。
    """

    def mark(match: re.Match[str]) -> str:
        return MATH_MARKER + split_markdown(match.group(0)).prose

    return split_markdown(
        "".join(
            segment if in_fence else MATH_SPAN.sub(mark, segment)
            for segment, in_fence in split_code_fences(markdown)
        )
    ).prose


def compare_pages(
    gemini_dir: Path, yomitoku_dir: Path, options: DiffOptions | None = None
) -> list[PageResult]:
    """2 つのキャッシュ（またはGemini は出力）ディレクトリの全ページを比較する。

    Args:
        gemini_dir: Gemini の出力 or ページキャッシュ（page_NNNN/raw.md）。
        yomitoku_dir: yomitoku のページキャッシュ（page_NNNN/analysis.json を含む）。

    Raises:
        FileNotFoundError: ディレクトリや、Gemini 側にあるページの analysis.json が無い場合。
        ValueError: analysis.json の形式が不正な場合。
    """
    results: list[PageResult] = []
    for page in list_prediction_pages(gemini_dir):
        analysis_path = yomitoku_dir / f"page_{page:04d}" / "analysis.json"
        if not analysis_path.is_file():
            raise FileNotFoundError(f"yomitoku の analysis.json が見つかりません: {analysis_path}")
        try:
            raw = json.loads(analysis_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{analysis_path} が JSON として読めません: {e}") from e
        analysis = parse_analysis(raw, source=str(analysis_path))
        results.append(compare_page(page, read_prediction(gemini_dir, page), analysis, options))
    return results


def _center_in(inner: BBox, outer: BBox) -> bool:
    """inner の中心が outer の内側にあるか。"""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _compare_paragraph(
    page: int,
    paragraph: _Paragraph,
    words: Sequence[_Word],
    prose: str,
    gemini_text: str,
    gemini_map: list[int],
    options: DiffOptions,
) -> tuple[list[Candidate], PageStats]:
    """yomitoku の段落 1 つを日本語の連続ごとに Gemini と比較する。"""
    text = unicodedata.normalize("NFKC", paragraph.contents).replace("\n", "")
    text = re.sub(r"\s+", "", text)
    comparable, index_map = build_comparable(text, fold_variants=options.normalize_variants)
    candidates: list[Candidate] = []
    stats = PageStats()
    offset = 0
    for run in comparable.split(SEPARATOR):
        run_start = offset
        offset += len(run) + 1
        if not run:
            continue
        stats = stats + PageStats(runs=1)
        if len(run) < options.min_run_length:
            stats = stats + PageStats(runs_short=1)
            continue
        if run in gemini_text:
            stats = stats + PageStats(runs_exact=1)
            continue
        context = _Context(page, paragraph, words, text, index_map, run_start, prose, gemini_map)
        found = _diff_run(run, gemini_text, context, options)
        if found is None:
            stats = stats + PageStats(runs_unmatched=1)
            span = _original_span(index_map, run_start, run_start + len(run), len(text))
            if options.report_unmatched:
                candidates.append(_make_candidate(context, "unmatched_run", span, None, options))
        else:
            stats = stats + PageStats(runs_different=1, ops_misaligned=found[1])
            candidates.extend(found[0])
    return candidates, stats


@dataclass(frozen=True)
class _Context:
    """1 つの run の位置情報（候補の文脈・座標を作るために持ち回る）。"""

    page: int
    paragraph: _Paragraph
    words: Sequence[_Word]
    text: str
    index_map: list[int]
    run_start: int
    prose: str
    gemini_map: list[int]


def _diff_run(
    run: str, gemini_text: str, context: _Context, options: DiffOptions
) -> tuple[list[Candidate], int] | None:
    """run と Gemini の対応箇所を文字単位で比較する。

    Returns:
        (誤読候補, 対応の取り違えとして除いた食い違いの件数)。対応箇所が見つからなければ None。
    """
    anchor = difflib.SequenceMatcher(None, gemini_text, run, autojunk=False).find_longest_match(
        0, len(gemini_text), 0, len(run)
    )
    if anchor.size < max(2, math.ceil(ANCHOR_MIN_RATIO * len(run))):
        return None
    start = max(0, anchor.a - anchor.b - WINDOW_PADDING)
    end = min(len(gemini_text), anchor.a + (len(run) - anchor.b) + WINDOW_PADDING)
    window = gemini_text[start:end]
    opcodes = difflib.SequenceMatcher(None, window, run, autojunk=False).get_opcodes()

    candidates: list[Candidate] = []
    misaligned = 0
    for position, (tag, a1, a2, b1, b2) in enumerate(opcodes):
        if tag == "equal":
            continue
        # 切り出しの余白は run の範囲外なので、両端の食い違いからは余白の分を除く
        if position == 0 and a1 == 0:
            if tag == "delete":
                continue
            a1 = _trim_leading_padding(window, a1, a2, b2 - b1)
        if position == len(opcodes) - 1 and a2 == len(window):
            if tag == "delete":
                continue
            a2 = _trim_trailing_padding(window, a1, a2, b2 - b1)
        gemini_part = window[a1:a2]
        yomitoku_part = run[b1:b2]
        if not yomitoku_part and not gemini_part.replace(SEPARATOR, ""):
            continue  # Gemini の数式の位置が、yomitoku では読まれていないだけ
        yomitoku_span = _original_span(
            context.index_map, context.run_start + b1, context.run_start + b2, len(context.text)
        )
        if _is_ignorable(gemini_part, yomitoku_part, options) or _is_symbol_noise(
            window, a1, a2, yomitoku_part, context.text[yomitoku_span[1] : yomitoku_span[1] + 1]
        ):
            continue
        if (
            gemini_part
            and yomitoku_part
            and max(len(gemini_part), len(yomitoku_part)) > MAX_REPLACE_LENGTH
        ):
            misaligned += 1
            continue
        if gemini_part and yomitoku_part:
            kind: CandidateKind = "replace"
        else:
            kind = "gemini_only" if gemini_part else "yomitoku_only"
        gemini_span = _original_span(context.gemini_map, start + a1, start + a2, len(context.prose))
        candidates.append(_make_candidate(context, kind, yomitoku_span, gemini_span, options))
    return candidates, misaligned


def _trim_leading_padding(window: str, a1: int, a2: int, yomitoku_length: int) -> int:
    """窓の先頭にある食い違いから、余白（run の範囲外）の分を除いた開始位置を返す。"""
    separator = window.rfind(SEPARATOR, a1, a2)
    if separator >= 0:
        a1 = separator + 1
    surplus = (a2 - a1) - yomitoku_length
    return a1 + min(max(surplus, 0), WINDOW_PADDING)


def _trim_trailing_padding(window: str, a1: int, a2: int, yomitoku_length: int) -> int:
    """窓の末尾にある食い違いから、余白（run の範囲外）の分を除いた終了位置を返す。"""
    separator = window.find(SEPARATOR, a1, a2)
    if separator >= 0:
        a2 = separator
    surplus = (a2 - a1) - yomitoku_length
    return a2 - min(max(surplus, 0), WINDOW_PADDING)


def _is_ignorable(gemini_part: str, yomitoku_part: str, options: DiffOptions) -> bool:
    """「ー」の揺れだけの食い違いか。

    yomitoku が「ー」だけを読んだ場合（数式のマイナスや分数線の読み違い）と、
    yomitoku が何も読んでいない側の Gemini が「ー」だけの場合（長音符の取りこぼし）を除く。
    """
    return (
        options.normalize_variants
        and set(yomitoku_part) <= _DASH_CHARS
        and (bool(yomitoku_part) or set(gemini_part) <= _DASH_CHARS)
    )


def _is_symbol_noise(window: str, a1: int, a2: int, yomitoku_part: str, yomitoku_next: str) -> bool:
    """数式・変数・図番号を yomitoku が仮名や記号として読んだだけの食い違いか。

    - Gemini の数式の位置（区切り）だけを置き換えた短い文字列、または数式の位置の隣への短い挿入
    - 段落に取り込まれた図番号・表番号の見出し語（直後が数字の「図」「表」）
    """
    if not yomitoku_part:
        return False
    if len(yomitoku_part) <= MAX_SYMBOL_NOISE_LENGTH:
        gemini_part = window[a1:a2]
        if gemini_part and set(gemini_part) == {SEPARATOR}:
            return True
        touches_gap = (a1 > 0 and window[a1 - 1] == SEPARATOR) or (
            a2 < len(window) and window[a2] == SEPARATOR
        )
        if not gemini_part and touches_gap:
            return True
    return yomitoku_part[-1] in _CAPTION_LABELS and yomitoku_next.isdigit()


def _original_span(
    index_map: list[int], start: int, end: int, original_length: int
) -> tuple[int, int]:
    """比較用文字列の範囲 [start, end) を元の文字列の範囲に直す。空の範囲は挿入位置を表す。"""
    if end > start:
        return index_map[start], index_map[end - 1] + 1
    position = index_map[start] if start < len(index_map) else original_length
    return position, position


def _make_candidate(
    context: _Context,
    kind: CandidateKind,
    yomitoku_span: tuple[int, int],
    gemini_span: tuple[int, int] | None,
    options: DiffOptions,
) -> Candidate:
    """食い違いの範囲から、文脈と座標つきの誤読候補を作る。"""
    n = options.context_chars
    yomitoku_text = context.text[yomitoku_span[0] : yomitoku_span[1]]
    if gemini_span is None:
        gemini_text = gemini_before = gemini_after = ""
    else:
        gemini_text = _readable(context.prose[gemini_span[0] : gemini_span[1]])
        gemini_before = _readable(context.prose[max(0, gemini_span[0] - n) : gemini_span[0]])
        gemini_after = _readable(context.prose[gemini_span[1] : gemini_span[1] + n])
    bbox, bbox_source = _locate(context, yomitoku_text)
    return Candidate(
        page=context.page,
        kind=kind,
        gemini_text=gemini_text,
        gemini_before=gemini_before,
        gemini_after=gemini_after,
        gemini_span=gemini_span,
        yomitoku_text=yomitoku_text,
        yomitoku_before=context.text[max(0, yomitoku_span[0] - n) : yomitoku_span[0]],
        yomitoku_after=context.text[yomitoku_span[1] : yomitoku_span[1] + n],
        bbox=bbox,
        bbox_source=bbox_source,
    )


def _readable(text: str) -> str:
    """数式の印を、出力で読める表記に置き換える。"""
    return text.replace(MATH_MARKER, MATH_LABEL)


def _locate(context: _Context, yomitoku_text: str) -> tuple[BBox, Literal["word", "paragraph"]]:
    """yomitoku 側の食い違い部分の座標を返す。単語（行）に収まれば単語の座標、そうでなければ段落の座標。"""
    if yomitoku_text:
        for word in context.words:
            xs = [p[0] for p in word.points]
            ys = [p[1] for p in word.points]
            word_box = (min(xs), min(ys), max(xs), max(ys))
            if not _center_in(word_box, context.paragraph.box):
                continue
            if yomitoku_text in unicodedata.normalize("NFKC", word.content).replace(" ", ""):
                return word_box, "word"
    return context.paragraph.box, "paragraph"
