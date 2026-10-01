"""仕様: OCR が出力したページ Markdown の表記揺れ・LaTeX 残骸を正規化する後処理。

OCR（主に Gemini）は書籍の右寄せ式番号を LaTeX 命令で再現しようとして、Markdown として
描画できない残骸を出力することがある。また、例・コメントなどのラベルや節見出しの
見出しレベルがページごとに揺れる。本モジュールはこれらを機械的に正規化する。

正規化ルール:
  - 数式外の \\hfill を除去する（「... \\hfill (1.18)」→「... (1.18)」）
  - 数式外の \\tag{X} を「(X)」に変換する
  - 本体が \\tag{X} だけの数式（$...$ / $$...$$ / equation 環境）を「(X)」に変換する
  - 数式外の \\quad / \\qquad を全角スペース 1 個 / 2 個に置き換える
  - 節番号（N.M）を持つ H3 以下の見出しを H2 に揃える
  - 見出し化された「例 N.M」「コメント N.M (C)」などのラベルを太字に戻す

注意事項:
  - コードフェンス内（図のテキスト表現など）は一切変更しない
  - H1 見出しは章検出（chapter_detect）の判定対象のため変更しない
  - 式本体を持つ数式の中身は変更しない
"""

from __future__ import annotations

import re

# コードフェンス（``` で囲まれたブロック）。中身は図のテキスト表現なので保護する
_CODE_FENCE = re.compile(r"^```[^\n]*\n[\s\S]*?^```[^\n]*$", re.MULTILINE)

# 数式スパン: ディスプレイ数式 / 数式外に置かれた LaTeX 環境 / インライン数式（\$ は除外）
_MATH_SPAN = re.compile(
    r"\$\$[\s\S]+?\$\$"
    r"|\\begin\{([a-zA-Z*]+)\}[\s\S]+?\\end\{\1\}"
    r"|(?<!\\)\$[^$\n]+?(?<!\\)\$"
)

# 本体が \tag{X} だけの数式（区切り記号・equation 環境を剥がした後の中身に対して判定する）
_TAG_ONLY_BODY = re.compile(r"^\s*\\tag\{([^{}]+)\}\s*$")
_MATH_DELIMITERS = re.compile(
    r"^(\$\$|\$|\\begin\{equation\*?\})([\s\S]*?)(\$\$|\$|\\end\{equation\*?\})$"
)

_HFILL = re.compile(r"[ \t]*\\hfill(?![a-zA-Z])[ \t]*")
_TAG = re.compile(r"\\tag\{([^{}]+)\}")
_QQUAD = re.compile(r"\\qquad(?![a-zA-Z])")
_QUAD = re.compile(r"\\quad(?![a-zA-Z])")

# 節番号 N.M で始まる H3 以下の見出し（N.M.K のような小節番号は対象外）
_SECTION_HEADING = re.compile(r"^#{3,}[ \t]*(\d+\.\d+(?:[ \t\u3000].*)?)$", re.MULTILINE)

# 見出し化されたラベル（例 4.4 / コメント 1.3 (C) / 練習 5.1 など）と、その後に続く本文
_LABEL_HEADING = re.compile(
    r"^#{3,}[ \t]*"
    r"((?:例|コメント|練習|定理|定義|命題|補題|系|問)[ \t]*\d+\.\d+(?:[ \t]*\([A-C]\))*)"
    r"(?:[ \t\u3000]+(.+?))?[ \t]*$",
    re.MULTILINE,
)

# 全角スペース（ラベルと後続文の区切り、\quad の置き換え先）
_FULLWIDTH_SPACE = "\u3000"


def _normalize_math_span(span: str) -> str:
    """本体が \\tag{X} だけの数式を式番号テキスト「(X)」に変換し、それ以外はそのまま返す。"""
    delimited = _MATH_DELIMITERS.match(span)
    if delimited is None:
        return span
    tag_only = _TAG_ONLY_BODY.match(delimited.group(2))
    if tag_only is None:
        return span
    return f"({tag_only.group(1)})"


def _normalize_prose(text: str) -> str:
    """数式外の地の文に残った LaTeX 命令を Markdown として読める表記に置き換える。"""
    text = _HFILL.sub(" ", text)
    text = _TAG.sub(r"(\1)", text)
    text = _QQUAD.sub(_FULLWIDTH_SPACE * 2, text)
    return _QUAD.sub(_FULLWIDTH_SPACE, text)


def _label_heading_to_bold(match: re.Match[str]) -> str:
    """見出し化されたラベルを「**ラベル**　後続文」の形に変換する。"""
    label, rest = match.group(1), match.group(2)
    if rest is None:
        return f"**{label}**"
    return f"**{label}**{_FULLWIDTH_SPACE}{rest}"


def _normalize_chunk(text: str) -> str:
    """コードフェンスを含まない Markdown 断片を正規化する。"""
    # 見出しは行単位で判定するため、数式スパンに分割する前に処理する
    text = _LABEL_HEADING.sub(_label_heading_to_bold, text)
    text = _SECTION_HEADING.sub(r"## \1", text)

    parts: list[str] = []
    last_end = 0
    for math in _MATH_SPAN.finditer(text):
        parts.append(_normalize_prose(text[last_end : math.start()]))
        parts.append(_normalize_math_span(math.group(0)))
        last_end = math.end()
    parts.append(_normalize_prose(text[last_end:]))
    return "".join(parts)


def normalize_ocr_markdown(markdown: str) -> str:
    """OCR 由来のページ Markdown に残る LaTeX 残骸と見出しレベルの揺れを正規化する。

    Args:
        markdown: 1 ページ分の Markdown 文字列。

    Returns:
        正規化後の Markdown 文字列。破綻が無ければ入力と同一の文字列を返す。
    """
    parts: list[str] = []
    last_end = 0
    for fence in _CODE_FENCE.finditer(markdown):
        parts.append(_normalize_chunk(markdown[last_end : fence.start()]))
        parts.append(fence.group(0))
        last_end = fence.end()
    parts.append(_normalize_chunk(markdown[last_end:]))
    return "".join(parts)
