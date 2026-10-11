"""仕様: searchable PDF のテキストレイヤーに、yomitoku の単語 bbox と別の OCR 結果の文字を組み合わせる（issue #42）。

Gemini は単語の位置（bbox）を返さないため、位置は yomitoku の単語 bbox を使い、文字は Gemini の
Markdown（人手修正・後処理の適用後）から取る。対応付けは次の手順で行う。

  1. yomitoku の analysis.json の単語を読み順に並べる（load_ordered_words）
  2. Gemini の Markdown を、記法を除き数式を平文に崩した文字列にする（markdown_to_search_text）
  3. 単語の文字を連結した文字列と Gemini の文字列を文字単位で突き合わせ（difflib）、
     各単語に Gemini 側の対応部分を割り当てる（assign_text）

割り当ての規則:
  - Gemini の文字は、どれか 1 つの単語にちょうど 1 回ずつ割り当てる（捨てない・重複させない）
  - yomitoku が読み落とした文字（Gemini にだけある文字）は、直後の単語に入れる。
    ページ末尾の文字は最後の単語に入れる
  - Gemini に対応する文字が無い単語（柱・ノンブルなど Gemini が書かない文字）は空にする。
    yomitoku の文字で補うと、Gemini 側で別の単語に入った文字と重複するため
  - 単語が 1 つも無いページ（図だけのページなど）の文字は描画できないため、置けなかった文字として数える
    （ページ単位で停止すると、OCR の後に書籍全体の PDF 出力が止まるため）
  - 件数（一致・置換・挿入・置けなかった文字数、空にした単語数）を AlignmentStats で返し、呼び出し側が報告する

注意事項:
  - テキストレイヤーの文字は NFKC 正規化し、空白を除いた文字列になる（英単語間の空白は残らない）
  - 数式の平文化は検索用の近似であり、LaTeX の意味を保存しない（例: x^2 → x2、\\frac{a}{b} → a/b）
  - yomitoku と Gemini の読み順が大きく異なる箇所では、文字が離れた単語の位置に入ることがある
"""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ouj_notebook_converter.evaluation.markdown_parts import split_markdown

BBox = tuple[int, int, int, int]

# LaTeX のトークン: 命令（\frac）/ 1 文字のエスケープ（\, や \\）/ 空白 / その他の 1 文字
_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+\*?|\\.|\s+|.", re.DOTALL)
# 文字として残さない 1 文字（区切り記号・上付き・下付き・表の列区切り）
_DROPPED_CHARS = frozenset("$^_&")
# 記号として残すエスケープ文字（\{ → {）
_ESCAPED_KEPT = frozenset("{}%$#&_|")
# 引数を文字として残さない命令と、読み飛ばす引数の数
_ARGUMENT_DROPPING = {
    "\\begin": 1,
    "\\end": 1,
    "\\hspace": 1,
    "\\hspace*": 1,
    "\\vspace": 1,
    "\\vspace*": 1,
    "\\label": 1,
    "\\enclose": 1,
    "\\cline": 1,
    "\\color": 1,
    "\\multicolumn": 2,
}
# 列指定 {rcl} を持つ環境（\begin{array}{rcl} の列指定も読み飛ばす）
_ENVIRONMENTS_WITH_COLUMN_SPEC = frozenset({"array", "tabular"})
_FRACTIONS = frozenset({"\\frac", "\\dfrac", "\\tfrac"})
_SIZED_DELIMITERS = frozenset({"\\left", "\\right", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr"})
# 記号の命令 → Unicode 文字
_SYMBOLS = {
    "\\times": "×",
    "\\cdot": "·",
    "\\div": "÷",
    "\\pm": "±",
    "\\mp": "∓",
    "\\leq": "≤",
    "\\le": "≤",
    "\\leqq": "≦",
    "\\geq": "≥",
    "\\ge": "≥",
    "\\geqq": "≧",
    "\\neq": "≠",
    "\\ne": "≠",
    "\\approx": "≈",
    "\\equiv": "≡",
    "\\sim": "∼",
    "\\cdots": "⋯",
    "\\ldots": "…",
    "\\dots": "…",
    "\\vdots": "⋮",
    "\\infty": "∞",
    "\\sqrt": "√",
    "\\to": "→",
    "\\rightarrow": "→",
    "\\leftarrow": "←",
    "\\Rightarrow": "⇒",
    "\\Leftrightarrow": "⇔",
    "\\iff": "⇔",
    "\\in": "∈",
    "\\notin": "∉",
    "\\subset": "⊂",
    "\\subseteq": "⊆",
    "\\cap": "∩",
    "\\cup": "∪",
    "\\emptyset": "∅",
    "\\forall": "∀",
    "\\exists": "∃",
    "\\angle": "∠",
    "\\triangle": "△",
    "\\perp": "⊥",
    "\\parallel": "∥",
    "\\circ": "°",
    "\\prime": "′",
    "\\sum": "Σ",
    "\\int": "∫",
    "\\alpha": "α",
    "\\beta": "β",
    "\\gamma": "γ",
    "\\delta": "δ",
    "\\epsilon": "ε",
    "\\varepsilon": "ε",
    "\\theta": "θ",
    "\\lambda": "λ",
    "\\mu": "μ",
    "\\pi": "π",
    "\\rho": "ρ",
    "\\sigma": "σ",
    "\\phi": "φ",
    "\\varphi": "φ",
    "\\omega": "ω",
    "\\Delta": "Δ",
    "\\Sigma": "Σ",
    "\\Omega": "Ω",
}
_WHITESPACE = re.compile(r"\s+")


def flatten_latex(span: str) -> str:
    """数式を、検索しやすい平文に崩す。

    区切り記号・上付き・下付き・波括弧・表の区切りを除き、記号の命令は Unicode 文字に、
    分数は「分子/分母」に、式番号 \\tag{X} は (X) にする。対応表に無い命令は名前を除き、
    引数の中身だけを残す。

    Args:
        span: 区切り記号（$ / $$ / \\( \\) / \\[ \\]）を含む数式、または LaTeX 環境。

    Returns:
        平文化した文字列（空白を含まない）。
    """
    tokens = [t for t in _LATEX_TOKEN.findall(span) if not t.isspace()]
    text, _ = _flatten_tokens(tokens, 0, until_close=False)
    return text


def _flatten_tokens(tokens: list[str], i: int, *, until_close: bool) -> tuple[str, int]:
    """tokens[i:] を平文化する。until_close なら対応する } の直後までを処理して返す。"""
    out: list[str] = []
    while i < len(tokens):
        token = tokens[i]
        if token == "}":
            if until_close:
                return "".join(out), i + 1
            i += 1  # 対応の無い閉じ括弧は文字として残さない
            continue
        if token == "{":
            group, i = _flatten_tokens(tokens, i + 1, until_close=True)
            out.append(group)
            continue
        if token in _ARGUMENT_DROPPING:
            i += 1
            environment = ""
            for _ in range(_ARGUMENT_DROPPING[token]):
                environment, i = _read_argument(tokens, i)
            if token == "\\begin" and environment in _ENVIRONMENTS_WITH_COLUMN_SPEC:
                _, i = _read_argument(tokens, i)
            continue
        if token in _FRACTIONS:
            numerator, i = _read_argument(tokens, i + 1)
            denominator, i = _read_argument(tokens, i)
            out.append(f"{numerator}/{denominator}")
            continue
        if token in ("\\tag", "\\tag*"):
            number, i = _read_argument(tokens, i + 1)
            out.append(f"({number})")
            continue
        if token in _SIZED_DELIMITERS:
            # \left. は見えない区切りのため何も残さない
            if i + 1 < len(tokens) and tokens[i + 1] == ".":
                i += 1
        elif token in _SYMBOLS:
            out.append(_SYMBOLS[token])
        elif token.startswith("\\"):
            if len(token) == 2 and token[1] in _ESCAPED_KEPT:
                out.append(token[1])
            # それ以外の命令（\text・\overline・\quad・\\ など）は名前を残さない
        elif token not in _DROPPED_CHARS:
            out.append(token)
        i += 1
    return "".join(out), i


def _read_argument(tokens: list[str], i: int) -> tuple[str, int]:
    """tokens[i] から始まる命令の引数 1 つ（{...} または 1 トークン）を平文化して返す。"""
    if i >= len(tokens):
        return "", i
    if tokens[i] == "{":
        return _flatten_tokens(tokens, i + 1, until_close=True)
    return _flatten_tokens(tokens[i : i + 1], 0, until_close=False)[0], i + 1


def markdown_to_search_text(markdown: str) -> str:
    """1 ページ分の Markdown を、テキストレイヤーに入れる文字列にする。

    評価の地の文（evaluation.markdown_parts.split_markdown）と同じ規則で Markdown 記法を除き、
    NFKC 正規化して空白を取り除く。数式は flatten_latex で平文に崩して含める。
    """
    return split_markdown(markdown, math_to_text=flatten_latex).prose


@dataclass(frozen=True)
class OcrWord:
    """yomitoku の単語（行）1 つ。

    Attributes:
        text: OCR した文字列。
        bbox: 画像座標（px）の外接矩形 (x1, y1, x2, y2)。
        direction: "horizontal" または "vertical"。
    """

    text: str
    bbox: BBox
    direction: str


def load_ordered_words(json_path: Path) -> list[OcrWord]:
    """yomitoku の analysis.json の単語を読み順に並べて返す。

    yomitoku の words は読み順に並んでいない（ページの下の行から並ぶことがある）。そこで、
    段落と図を order 順に並べ、図はその中の段落を order 順に展開したうえで、各単語を中心が
    含まれる最初の段落に入れる。段落の中は、横書きなら上の行から、縦書きなら右の列から並べる。
    どの段落にも入らない単語（柱・ノンブルなど）は最後に上から順に並べる。
    文字か座標の無い単語は描画できないため除く。

    Args:
        json_path: yomitoku の analysis.json のパス。

    Returns:
        読み順の単語のリスト。
    """
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    blocks = _ordered_blocks(data)
    buckets: list[list[OcrWord]] = [[] for _ in blocks]
    rest: list[OcrWord] = []
    for raw in data.get("words", []):
        content = raw.get("content")
        points = raw.get("points")
        if not content or not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        word = OcrWord(
            text=content,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            direction=raw.get("direction", "horizontal"),
        )
        center_x = (word.bbox[0] + word.bbox[2]) / 2
        center_y = (word.bbox[1] + word.bbox[3]) / 2
        for index, block in enumerate(blocks):
            x1, y1, x2, y2 = block["box"]
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                buckets[index].append(word)
                break
        else:
            rest.append(word)

    ordered: list[OcrWord] = []
    for block, words in zip(blocks, buckets, strict=True):
        if block.get("direction") == "vertical":
            ordered.extend(sorted(words, key=lambda w: (-w.bbox[2], w.bbox[1])))
        else:
            ordered.extend(sorted(words, key=lambda w: (w.bbox[1], w.bbox[0])))
    ordered.extend(sorted(rest, key=lambda w: (w.bbox[1], w.bbox[0])))
    return ordered


def _ordered_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """段落と図の中の段落を読み順に並べる。図の中の段落に入らない図中の単語のため、図そのものも末尾に加える。"""
    items: list[tuple[float, list[dict[str, Any]]]] = [
        (_order(p), [p]) for p in data.get("paragraphs", [])
    ]
    for figure in data.get("figures", []):
        inner = sorted(figure.get("paragraphs", []), key=_order)
        items.append((_order(figure), [*inner, figure]))
    items.sort(key=lambda item: item[0])
    return [block for _, blocks in items for block in blocks]


def _order(block: dict[str, Any]) -> float:
    order = block.get("order")
    return math.inf if order is None else float(order)


@dataclass(frozen=True)
class AlignmentStats:
    """単語への文字の割り当て件数。

    Attributes:
        words: 単語数。
        words_without_text: 割り当てる文字が無く空にした単語数。
        chars: 割り当てた文字数（Gemini 側の文字列の長さ）。
        chars_matched: yomitoku と一致した文字数。
        chars_replaced: yomitoku の異なる文字の位置に割り当てた文字数。
        chars_inserted: yomitoku に対応する文字が無く、隣の単語に入れた文字数。
        chars_unplaced: ページに単語が 1 つも無く、描画できなかった文字数。
    """

    words: int
    words_without_text: int
    chars: int
    chars_matched: int
    chars_replaced: int
    chars_inserted: int
    chars_unplaced: int = 0

    def __add__(self, other: AlignmentStats) -> AlignmentStats:
        return AlignmentStats(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )


@dataclass(frozen=True)
class TextAssignment:
    """assign_text の結果。texts[i] は i 番目の単語に描画する文字列。"""

    texts: tuple[str, ...]
    stats: AlignmentStats


def assign_text(word_texts: Sequence[str], text: str) -> TextAssignment:
    """yomitoku の単語ごとに、text のうち対応する部分を割り当てる。

    単語の文字列（NFKC 正規化・空白除去）を連結したものと text を文字単位で突き合わせ、
    yomitoku の各文字位置に対応する text の位置を求めて、単語の範囲で text を切り分ける。
    割り当ての規則はモジュールの説明を参照。

    Args:
        word_texts: 読み順に並んだ yomitoku の単語の文字列。
        text: 割り当てる文字列（markdown_to_search_text の戻り値）。

    Returns:
        単語ごとの文字列と件数。単語が 1 つも無い場合、text の文字はすべて chars_unplaced に数える。
    """
    if not word_texts:
        return TextAssignment(
            texts=(),
            stats=AlignmentStats(
                words=0,
                words_without_text=0,
                chars=len(text),
                chars_matched=0,
                chars_replaced=0,
                chars_inserted=0,
                chars_unplaced=len(text),
            ),
        )

    normalized = [_WHITESPACE.sub("", unicodedata.normalize("NFKC", t)) for t in word_texts]
    joined = "".join(normalized)
    matcher = difflib.SequenceMatcher(None, joined, text, autojunk=False)

    # starts[k]: yomitoku の k 文字目に割り当てる text の開始位置（単調非減少）
    starts = [0] * (len(joined) + 1)
    insertions: list[tuple[int, int]] = []
    matched = replaced = inserted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += j2 - j1
            for k in range(i1, i2):
                starts[k] = j1 + (k - i1)
        elif tag == "replace":
            replaced += j2 - j1
            for k in range(i1, i2):
                starts[k] = j1 + (k - i1) * (j2 - j1) // (i2 - i1)
        elif tag == "delete":
            for k in range(i1, i2):
                starts[k] = j1
        else:
            inserted += j2 - j1
            insertions.append((i1, j1))
    # 挿入された文字は直後の単語（yomitoku の i1 文字目を含む単語）に入れる。
    # 直後の操作が starts[i1] を上書きするため、すべての操作の後で反映する
    for i1, j1 in insertions:
        if i1 < len(joined):
            starts[i1] = j1
    starts[0] = 0
    starts[len(joined)] = len(text)

    texts: list[str] = []
    position = 0
    for word in normalized:
        end = position + len(word)
        texts.append(text[starts[position] : starts[end]])
        position = end

    return TextAssignment(
        texts=tuple(texts),
        stats=AlignmentStats(
            words=len(texts),
            words_without_text=sum(1 for t in texts if not t),
            chars=len(text),
            chars_matched=matched,
            chars_replaced=replaced,
            chars_inserted=inserted,
        ),
    )
