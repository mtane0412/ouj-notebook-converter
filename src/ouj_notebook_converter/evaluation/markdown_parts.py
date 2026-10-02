"""仕様: 評価のため、1 ページ分の Markdown を「地の文」「数式」「見出し」に分解する。

- 地の文: 数式と Markdown 記法を除き、NFKC 正規化して空白を取り除いた文字列（CER の比較対象）
- 数式: 出現順の数式（区切り記号を除いた本体とディスプレイ数式かどうか）
- 見出し: レベルとテキストの組

注意事項:
  - 数式の検出には変換時の後処理（markdown_cleanup）と同じ正規表現を使う
  - NFKC 正規化により全角・半角の違い（「，」と「,」など）は誤りとして数えない
  - コードフェンス内の行は見出しとして扱わないが、地の文には含める（図中の文字も本文の一部とみなす）
  - 数式内の \\text{...} の中身と \\tag{X}（「(X)」として）は地の文として扱う。OCR が日本語や
    式番号を数式の中に入れても外に出しても、同じ地の文・同じ数式として比較するため
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN


@dataclass(frozen=True)
class Formula:
    """Markdown 中の 1 つの数式。

    Attributes:
        tex: 区切り記号（$ / $$ / \\( \\) / \\[ \\]）を除いた数式本体。LaTeX 環境は環境ごと保持する。
        display: ディスプレイ数式なら True。
    """

    tex: str
    display: bool


@dataclass(frozen=True)
class Heading:
    """Markdown の見出し。

    Attributes:
        level: 見出しレベル（# の個数）。
        text: 強調記号と空白を除き、NFKC 正規化した見出しテキスト。
    """

    level: int
    text: str


@dataclass(frozen=True)
class MarkdownParts:
    """split_markdown の結果。"""

    prose: str
    formulas: tuple[Formula, ...]
    headings: tuple[Heading, ...]


_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t#]*$")
_CODE_FENCE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})")
# 表の区切り行（| --- | :---: | など）
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|?)+\s*$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)[ \t]+")
_BLOCKQUOTE = re.compile(r"^\s*>+[ \t]?")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<!--[\s\S]*?-->|<[^>\n]+>")
_EMPHASIS = re.compile(r"\*+|_{2,}")
# バックスラッシュエスケープされた ASCII 記号（\* → *）
_ESCAPED_PUNCTUATION = re.compile(r"\\([!-/:-@\[-`{-~])")
_WHITESPACE = re.compile(r"\s+")
# 数式内の地の文: \text{...} 系の命令と式番号 \tag{...}
_PROSE_IN_MATH = re.compile(r"\\(?:text|textrm|mbox)\{([^{}]*)\}|\\tag\*?\{([^{}]*)\}")

# LaTeX のトークン: 命令（\frac）/ 1 文字のエスケープ（\, や \\）/ 空白
_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\s+", re.DOTALL)
# 表示に影響しない、または表示がほぼ同じになるため比較時に取り除く命令
_LATEX_DROPPED = frozenset(
    {
        "\\,",
        "\\;",
        "\\:",
        "\\!",
        "\\ ",
        "\\quad",
        "\\qquad",
        "\\left",
        "\\right",
        "\\displaystyle",
        "\\textstyle",
    }
)
# 同じ表示になる命令の別名
_LATEX_ALIASES = {
    "\\dfrac": "\\frac",
    "\\tfrac": "\\frac",
    "\\le": "\\leq",
    "\\ge": "\\geq",
    "\\ne": "\\neq",
}
_TRAILING_PUNCTUATION = re.compile(r"[.,。、]+$")


def split_markdown(markdown: str) -> MarkdownParts:
    """1 ページ分の Markdown を地の文・数式・見出しに分解する。

    Args:
        markdown: 1 ページ分の Markdown 文字列。

    Returns:
        分解結果。
    """
    formulas = tuple(_to_formula(match.group(0)) for match in MATH_SPAN.finditer(markdown))
    # 数式を「数式内の地の文」に置き換えてから行単位で記法を除く（数式は複数行にまたがりうるため先に除く）
    without_math = MATH_SPAN.sub(lambda m: f" {_prose_in_math(m.group(0))} ", markdown)

    prose_lines: list[str] = []
    headings: list[Heading] = []
    in_fence = False
    for line in without_math.splitlines():
        if _CODE_FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            heading = _HEADING.match(line)
            if heading is not None:
                headings.append(
                    Heading(level=len(heading.group(1)), text=_normalize_text(heading.group(2)))
                )
                line = heading.group(2)
        prose_lines.append(_strip_markup(line))

    return MarkdownParts(
        prose=_normalize_text("".join(prose_lines)),
        formulas=formulas,
        headings=tuple(headings),
    )


def normalize_latex(tex: str) -> str:
    """数式を比較するため、表示に影響しない表記の違いを取り除いた LaTeX 文字列を返す。

    地の文として扱う \\text{...}・\\tag{...}、空白・間隔調整命令・\\left/\\right・末尾の句読点を除き、
    \\dfrac→\\frac などの別名を統一する。日本語だけの数式は空文字列になる。
    根指数や添字など数学的な意味が変わる違いは残す。

    Args:
        tex: 区切り記号を除いた数式本体。

    Returns:
        正規化後の文字列（比較専用。LaTeX として描画できるとは限らない）。
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isspace() or token in _LATEX_DROPPED:
            return ""
        return _LATEX_ALIASES.get(token, token)

    without_prose = _PROSE_IN_MATH.sub("", unicodedata.normalize("NFKC", tex))
    normalized = _LATEX_TOKEN.sub(replace, without_prose)
    return _TRAILING_PUNCTUATION.sub("", normalized)


def _to_formula(span: str) -> Formula:
    """MATH_SPAN に一致した文字列から区切り記号を除いて Formula を作る。"""
    if span.startswith("$$"):
        return Formula(tex=span[2:-2].strip(), display=True)
    if span.startswith("\\["):
        return Formula(tex=span[2:-2].strip(), display=True)
    if span.startswith("\\("):
        return Formula(tex=span[2:-2].strip(), display=False)
    if span.startswith("\\begin"):
        return Formula(tex=span, display=True)
    return Formula(tex=span[1:-1].strip(), display=False)


def _prose_in_math(span: str) -> str:
    """数式中の \\text{...} の中身と \\tag{X}（「(X)」として）を出現順に連結して返す。"""
    return " ".join(
        m.group(1) if m.group(1) is not None else f"({m.group(2)})"
        for m in _PROSE_IN_MATH.finditer(span)
    )


def _strip_markup(line: str) -> str:
    """1 行から Markdown 記法を取り除く。"""
    if _TABLE_SEPARATOR.match(line):
        return ""
    line = _BLOCKQUOTE.sub("", line)
    line = _LIST_MARKER.sub("", line)
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _HTML.sub("", line)
    line = _EMPHASIS.sub("", line)
    line = _ESCAPED_PUNCTUATION.sub(r"\1", line)
    return line.replace("|", " ")


def _normalize_text(text: str) -> str:
    """NFKC 正規化し、強調記号と空白を取り除く。"""
    text = _EMPHASIS.sub("", unicodedata.normalize("NFKC", text))
    return _WHITESPACE.sub("", text)
