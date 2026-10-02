"""仕様: OCR 結果の数式に含まれる等式・不等式を記号計算で検算し、成り立たない式を誤読候補として検出する。

issue #19 のプロトタイプ。例: p.70 で 4 乗根を 3 乗根と誤読した `(\\sqrt[3]{a})^3 = (\\sqrt[3]{a})^4` は
恒等的に成り立たないため、画像を見ずに検出できる。

処理の流れ:
  1. extract_links: 数式を文（カンマ・\\quad・\\text{...}・\\Rightarrow などで区切る）に分け、
     文の中の関係記号（= < > \\leq \\geq \\neq など）の連鎖を、隣り合う 2 項ずつのリンクに分解する
  2. check_link: 両辺を sympy の LaTeX パーサー（lark バックエンド）で式にし、
     文字に標本値を代入した数値計算で成否を判定する
  3. check_markdown: 1 ページ分の Markdown の全数式を検算する

判定の規則（LinkStatus）:
  - どの標本値でも成り立つ → HOLDS
  - どの標本値でも成り立たない → VIOLATED（誤読候補）。ただし次は誤読候補にしない
      * 文字を含む等式で、片辺が文字 1 つか数値 1 つ（定義・解・「= 0」の方程式）、
        または両辺の文字の組が異なる（方程式） → CONDITIONAL
      * 直後の文が「〜ではない」「〜としてはいけない」のように否定している → NEGATED
  - 標本値によって成否が変わる（条件付きの不等式など） → CONDITIONAL
  - 日本語を含む・空・解釈できない・数値計算できない項 → SKIPPED_*（件数だけ記録する）
  - 数値計算が LINK_TIMEOUT_SECONDS を超えたリンク → SKIPPED_TIMEOUT（総和の上限 n に小数を
    代入すると sympy の評価が終わらないことがあるため打ち切る）

注意事項:
  - 標本値は正負の実数から固定シードの乱数で選ぶため、結果は実行ごとに変わらない
  - 文字 i を虚数単位と読み直して成り立つなら HOLDS とする（複素数の章で i を虚数単位に使うため）
  - lark バックエンドは \\pi を解釈できないため、記号 pi として読んでから円周率に置き換える
  - 筆算・場合分け・行列の環境（array, cases など）を含む数式は検算の対象外とする
  - 時間制限に SIGALRM を使うため、check_link は Unix のメインスレッドからだけ呼び出せる
  - sympy と lark は任意依存（verify extra）。未インストールの場合は import 時に ImportError で停止する
"""

from __future__ import annotations

import cmath
import random
import re
import signal
import unicodedata
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from itertools import product
from types import FrameType

import sympy
from lark import Tree
from lark.exceptions import LarkError
from sympy.parsing.latex import parse_latex
from sympy.utilities.exceptions import SymPyDeprecationWarning

from ouj_notebook_converter.evaluation.markdown_parts import split_markdown
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN, split_code_fences


class LinkStatus(str, Enum):
    """リンク（左辺・関係記号・右辺の組）の検算結果。"""

    HOLDS = "holds"
    """どの標本値でも成り立つ。"""
    VIOLATED = "violated"
    """どの標本値でも成り立たない。誤読候補。"""
    CONDITIONAL = "conditional"
    """条件付きで成り立つ（方程式・定義・条件付きの不等式）。誤読かどうか判断しない。"""
    NEGATED = "negated"
    """成り立たないが、直後の文で否定されている（「〜ではない」など）。"""
    SKIPPED_NON_ASCII = "skipped_non_ascii"
    """項に日本語など ASCII 以外の文字が含まれるため検算しない。"""
    SKIPPED_EMPTY = "skipped_empty"
    """項が空（数式が関係記号で始まる・終わる）のため検算しない。"""
    SKIPPED_PARSE = "skipped_parse"
    """項を LaTeX の式として解釈できない。"""
    SKIPPED_EVAL = "skipped_eval"
    """どの標本値でも数値計算できない（未定義関数・極限・総和など）。"""
    SKIPPED_TIMEOUT = "skipped_timeout"
    """数値計算が時間制限を超えたため打ち切った。"""


@dataclass(frozen=True)
class Link:
    """関係式の連鎖のうち、隣り合う 2 項と関係記号の組。

    Attributes:
        lhs: 左辺の LaTeX。
        relation: 関係記号（"=", "<", ">", "\\leq" など。数式中の表記のまま）。
        rhs: 右辺の LaTeX。
        following_text: このリンクを含む文の直後にある文（\\text{...} の中身や数式の後ろの地の文）。
            否定（「〜ではない」）の判定に使う。
    """

    lhs: str
    relation: str
    rhs: str
    following_text: str = ""


@dataclass(frozen=True)
class LinkResult:
    """check_markdown が返す、1 リンクの検算結果。

    Attributes:
        formula_index: リンクを含む数式の、ページ内での出現順の番号（0 始まり。
            evaluation.markdown_parts.split_markdown の formulas の添字と一致する）。
        formula: リンクを含む数式の本体（区切り記号を除いた LaTeX）。
        link: リンク。
        status: 検算結果。
    """

    formula_index: int
    formula: str
    link: Link
    status: LinkStatus


@dataclass(frozen=True)
class MarkdownCheck:
    """check_markdown の結果。

    Attributes:
        links: ページ内の全リンクの検算結果（数式の出現順）。
        formula_count: ページ内の数式の件数。
        skipped_formula_count: 筆算・場合分けなど対象外の環境を含むため検算しなかった数式の件数。
    """

    links: tuple[LinkResult, ...]
    formula_count: int
    skipped_formula_count: int


# 関係記号として連鎖を分ける命令（数式中の表記 → 比較の種類）
_RELATION_COMMANDS = {
    "\\le": "<=",
    "\\leq": "<=",
    "\\leqq": "<=",
    "\\leqslant": "<=",
    "\\ge": ">=",
    "\\geq": ">=",
    "\\geqq": ">=",
    "\\geqslant": ">=",
    "\\ne": "!=",
    "\\neq": "!=",
    "\\lt": "<",
    "\\gt": ">",
}
_RELATION_CHARS = {"=": "=", "<": "<", ">": ">"}
# 文を区切る命令。前後の式は連鎖としてつながない（論理記号・集合・近似・極限の矢印など）
_BREAK_COMMANDS = frozenset(
    {
        "\\quad",
        "\\qquad",
        "\\Rightarrow",
        "\\Leftarrow",
        "\\Leftrightarrow",
        "\\Longrightarrow",
        "\\Longleftarrow",
        "\\Longleftrightarrow",
        "\\iff",
        "\\implies",
        "\\therefore",
        "\\because",
        "\\to",
        "\\rightarrow",
        "\\in",
        "\\notin",
        "\\ni",
        "\\subset",
        "\\subseteq",
        "\\supset",
        "\\supseteq",
        "\\approx",
        "\\fallingdotseq",
        "\\risingdotseq",
        "\\simeq",
        "\\equiv",
        "\\sim",
        "\\mapsto",
    }
)
_BREAK_CHARS = frozenset({",", ";"})
_TEXT_COMMANDS = frozenset({"\\text", "\\textrm", "\\mbox"})
_OPEN_BRACKETS = frozenset({"(", "[", "{"})
_CLOSE_BRACKETS = frozenset({")", "]", "}"})

_UNSUPPORTED_ENVIRONMENT = re.compile(
    r"\\begin\{(?:array|cases|tabular|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\*?\}"
)
_ENVIRONMENT = re.compile(r"\\(?:begin|end)\{[A-Za-z]+\*?\}")
_TAG = re.compile(r"\\tag\*?\{[^{}]*\}")
_LINE_BREAK = re.compile(r"\\\\(?:\[[^\]]*\])?")
_DROPPED = re.compile(
    r"\\(?:left|right|middle|displaystyle|textstyle|bigl|bigr|Bigl|Bigr|big|Big)(?![A-Za-z])"
    r"|\\[,;:! ]|~|&"
)
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|.", re.DOTALL)
_TEXT_ARGUMENT = re.compile(r"\{([^{}]*)\}")

# 項の解釈前の書き換え
_DEGREE = re.compile(r"(\d+(?:\.\d+)?)\s*\^\s*(?:\\circ|\{\s*\\circ\s*\})")
_PI = re.compile(r"\\pi(?![A-Za-z])")
_MATHRM = re.compile(r"\\mathrm\{([^{}]*)\}")
_DFRAC = re.compile(r"\\[dt]frac(?![A-Za-z])")

# 片辺が文字 1 つ（添字付きを含む）か数値 1 つかの判定
_TRIVIAL_TERM = re.compile(
    r"\s*-?\s*(?:\d+(?:\.\d+)?|[A-Za-z]|\\[A-Za-z]+)(?:_\{?[A-Za-z0-9]+\}?)?\s*"
)
# 改行や \\cdots で切れた式の断片（+ で始まる・演算子で終わる）
_FRAGMENT = re.compile(r"^\s*\+|(?:[+\-*/]|\\cdot|\\times|\\div)\s*$")
_DECIMAL_TERM = re.compile(r"\s*-?\s*\d+\.(\d+)\s*")
# 直後の文が否定している（「ではない」「としてはいけない」「とはなりません」など）
_NEGATION = re.compile(r"^[^。．.]{0,12}?(?:ない|ません)")

_PI_SYMBOL = sympy.Symbol("pi")
_I_SYMBOL = sympy.Symbol("i")
# 標本値の候補。0・1・整数を避け、条件付きの等式（x = 2 など）が偶然成り立たないようにする。
# 範囲の条件（a \\le -3, x > 100 など）が成り立つ値も含むよう、正負に広く取る
_SAMPLE_POOL = (0.13, 0.37, 0.71, 0.93, 1.3, 1.9, 2.6, 5.2, 11.3, 101.7)
_SAMPLES = (*_SAMPLE_POOL, *(-v for v in _SAMPLE_POOL))
# 総和・総乗の範囲（\\sum_{k=1}^n の n など）に使う文字に代入する整数
_INTEGER_SAMPLES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
_SAMPLE_SEED = 19
_RELATIVE_TOLERANCE = 1e-9
LINK_TIMEOUT_SECONDS = 2.0
"""リンク 1 件の解釈と数値計算にかける時間の上限（秒）。"""


class _LinkTimeoutError(Exception):
    """リンクの検算が時間制限を超えたことを知らせる内部用の例外。"""


def extract_links(tex: str, following_text: str = "") -> list[Link] | None:
    """数式の本体から、関係式の連鎖を隣り合う 2 項ずつのリンクに分解する。

    Args:
        tex: 区切り記号（$ など）を除いた数式の本体。
        following_text: 数式の直後の地の文。数式の最後の文の following_text になる。

    Returns:
        リンクの一覧（数式中の出現順）。筆算・場合分け・行列などの環境を含む数式は None。
    """
    if _UNSUPPORTED_ENVIRONMENT.search(tex):
        return None
    body = _TAG.sub(" ", _ENVIRONMENT.sub(" ", tex))
    links: list[Link] = []
    statements = _split_statements(_join_continued_lines(body))
    for index, (statement, text_after) in enumerate(statements):
        if index == len(statements) - 1:
            text_after += following_text
        terms, relations = _split_chain(statement)
        for i, relation in enumerate(relations):
            links.append(
                Link(
                    lhs=terms[i],
                    relation=relation,
                    rhs=terms[i + 1],
                    following_text=text_after,
                )
            )
    return links


def check_link(link: Link) -> LinkStatus:
    """リンクを記号計算で検算する。

    Args:
        link: 検算するリンク。

    Returns:
        検算結果。判定の規則はモジュールの docstring を参照。

    Raises:
        ValueError: メインスレッド以外から呼び出した場合（SIGALRM を設定できないため）。
    """
    return _with_time_limit(lambda: _check_link(link))


def _with_time_limit(check: Callable[[], LinkStatus]) -> LinkStatus:
    """check を LINK_TIMEOUT_SECONDS 以内で実行し、超えたら SKIPPED_TIMEOUT を返す。"""

    def on_alarm(_signum: int, _frame: FrameType | None) -> None:
        raise _LinkTimeoutError

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, LINK_TIMEOUT_SECONDS)
    try:
        return check()
    except _LinkTimeoutError:
        return LinkStatus.SKIPPED_TIMEOUT
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _check_link(link: Link) -> LinkStatus:
    """check_link の本体（時間制限なし）。"""
    if not link.lhs.strip() or not link.rhs.strip():
        return LinkStatus.SKIPPED_EMPTY
    if not (link.lhs + link.rhs).isascii():
        return LinkStatus.SKIPPED_NON_ASCII
    if _is_fragment(link.lhs) or _is_fragment(link.rhs) or "'" in link.lhs + link.rhs:
        # 断片は式として不完全、f'(x) は未定義関数の微分で数値計算できない
        return LinkStatus.SKIPPED_PARSE
    lhs_candidates = _parse_term(link.lhs)
    rhs_candidates = _parse_term(link.rhs)
    if not lhs_candidates or not rhs_candidates:
        return LinkStatus.SKIPPED_PARSE

    comparison = _comparison(link.relation)
    tolerance = max(_decimal_tolerance(link.lhs), _decimal_tolerance(link.rhs))
    outcomes: list[LinkStatus] = []
    has_symbols = False
    symbol_sets_differ = False
    # 2 通りに読める式（f(x) は関数適用か積か）は、組み合わせのどれかで成り立てば成り立つとみなす
    for lhs, rhs in product(lhs_candidates, rhs_candidates):
        has_symbols |= bool(lhs.free_symbols | rhs.free_symbols)
        symbol_sets_differ |= lhs.free_symbols != rhs.free_symbols
        outcomes.extend(_evaluate_variants(lhs, rhs, comparison, tolerance))

    if not outcomes:
        return LinkStatus.SKIPPED_EVAL
    if LinkStatus.HOLDS in outcomes:
        return LinkStatus.HOLDS
    if LinkStatus.CONDITIONAL in outcomes:
        return LinkStatus.CONDITIONAL
    # ここに来るのは、どの読み方・どの標本値でも成り立たない場合
    if (
        comparison == "="
        and has_symbols
        and (
            _TRIVIAL_TERM.fullmatch(link.lhs)
            or _TRIVIAL_TERM.fullmatch(link.rhs)
            or symbol_sets_differ
        )
    ):
        return LinkStatus.CONDITIONAL
    if _NEGATION.search(unicodedata.normalize("NFKC", link.following_text).strip()):
        return LinkStatus.NEGATED
    return LinkStatus.VIOLATED


def check_markdown(markdown: str) -> MarkdownCheck:
    """1 ページ分の Markdown に含まれる全数式のリンクを検算する。

    数式の切り出しと出現順は evaluation.markdown_parts.split_markdown と同じ規則に従う。

    Args:
        markdown: 1 ページ分の Markdown（変換時の後処理を適用済みのもの）。

    Returns:
        検算結果。
    """
    formulas = split_markdown(markdown).formulas
    following_texts = list(_following_texts(markdown))
    if len(formulas) != len(following_texts):
        raise ValueError(
            f"数式の件数が一致しません（split_markdown: {len(formulas)} 件, "
            f"数式の後ろの地の文: {len(following_texts)} 件）"
        )
    results: list[LinkResult] = []
    skipped = 0
    for index, (formula, following) in enumerate(zip(formulas, following_texts, strict=True)):
        links = extract_links(formula.tex, following_text=following)
        if links is None:
            skipped += 1
            continue
        results.extend(
            LinkResult(formula_index=index, formula=formula.tex, link=link, status=check_link(link))
            for link in links
        )
    return MarkdownCheck(
        links=tuple(results), formula_count=len(formulas), skipped_formula_count=skipped
    )


def _following_texts(markdown: str) -> Iterator[str]:
    """数式ごとに、数式の直後から行末までの地の文を返す（コードフェンス内の数式は除く）。"""
    for segment, in_fence in split_code_fences(markdown):
        if in_fence:
            continue
        for match in MATH_SPAN.finditer(segment):
            rest = segment[match.end() :]
            yield rest.split("\n", 1)[0]


def _join_continued_lines(body: str) -> str:
    """改行（\\\\）で行に分け、関係記号で始まる行（aligned の継続行）は前の行につなげる。

    つながらない行の境目には文の区切り（;）を入れる。
    """
    lines = [_DROPPED.sub(" ", line).strip() for line in _LINE_BREAK.split(body)]
    joined: list[str] = []
    for line in lines:
        if not line:
            continue
        if joined and _starts_with_relation(line):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return " ; ".join(joined)


def _starts_with_relation(line: str) -> bool:
    first = _TOKEN.match(line)
    return first is not None and (
        first.group(0) in _RELATION_CHARS or first.group(0) in _RELATION_COMMANDS
    )


def _split_statements(body: str) -> list[tuple[str, str]]:
    """括弧の外にある区切り（カンマ・\\quad・\\text{...} など）で文に分ける。

    Returns:
        (文, 文の直後の区切りに含まれる文字列) の組の一覧。空の文は含めず、
        区切りが続く場合（\\quad \\text{となり} など）はその文字列を直前の文の直後の文字列に連結する。
    """
    statements: list[tuple[str, str]] = []
    current: list[str] = []
    depth = 0
    position = 0
    while position < len(body):
        token = _TOKEN.match(body, position)
        assert token is not None  # _TOKEN は任意の 1 文字に一致する
        value = token.group(0)
        position = token.end()
        is_break = False
        text = ""
        if depth <= 0 and value in _TEXT_COMMANDS:
            argument = _TEXT_ARGUMENT.match(body, position)
            if argument is not None:
                is_break, text = True, argument.group(1)
                position = argument.end()
        elif depth <= 0 and (value in _BREAK_COMMANDS or value in _BREAK_CHARS):
            is_break = True
        if is_break:
            statement = "".join(current).strip()
            if statement:
                statements.append((statement, text))
            elif statements:
                previous, previous_text = statements[-1]
                statements[-1] = (previous, previous_text + text)
            current = []
            continue
        if value in _OPEN_BRACKETS or value in ("\\{", "\\lbrace"):
            depth += 1
        elif value in _CLOSE_BRACKETS or value in ("\\}", "\\rbrace"):
            depth -= 1
        current.append(value)
    statement = "".join(current).strip()
    if statement:
        statements.append((statement, ""))
    return statements


def _split_chain(statement: str) -> tuple[list[str], list[str]]:
    """文を括弧の外の関係記号で分け、項の一覧と関係記号の一覧を返す。"""
    terms: list[str] = []
    relations: list[str] = []
    current: list[str] = []
    depth = 0
    for token in _TOKEN.finditer(statement):
        value = token.group(0)
        if depth <= 0 and (value in _RELATION_CHARS or value in _RELATION_COMMANDS):
            terms.append("".join(current).strip())
            relations.append(value)
            current = []
            continue
        if value in _OPEN_BRACKETS or value in ("\\{", "\\lbrace"):
            depth += 1
        elif value in _CLOSE_BRACKETS or value in ("\\}", "\\rbrace"):
            depth -= 1
        current.append(value)
    terms.append("".join(current).strip())
    return terms, relations


def _is_fragment(term: str) -> bool:
    return _FRAGMENT.search(term) is not None


def _comparison(relation: str) -> str:
    if relation in _RELATION_CHARS:
        return _RELATION_CHARS[relation]
    if relation in _RELATION_COMMANDS:
        return _RELATION_COMMANDS[relation]
    raise ValueError(f"未対応の関係記号です: {relation}")


def _parse_term(term: str) -> list[sympy.Expr]:
    """項を sympy の式にする。2 通りに読める場合（f(x) など）は両方を返し、解釈できなければ空を返す。"""
    tex = _DROPPED.sub(" ", term)
    tex = _DEGREE.sub(r"(\1 \\cdot \\pi / 180)", tex)
    tex = _PI.sub(r"\\mathit{pi}", tex)
    tex = _MATHRM.sub(r"\1", tex)
    tex = _DFRAC.sub(r"\\frac", tex)
    try:
        with warnings.catch_warnings():
            # lark バックエンドの変換器が古い sympy API を使うことによる警告。判定には影響しない
            warnings.simplefilter("ignore", SymPyDeprecationWarning)
            parsed = parse_latex(tex, backend="lark")
    except (LarkError, TypeError, ValueError, AttributeError):
        return []
    if isinstance(parsed, Tree) and parsed.data == "_ambig":
        candidates = [child for child in parsed.children if isinstance(child, sympy.Expr)]
    elif isinstance(parsed, sympy.Expr):
        candidates = [parsed]
    else:
        # 関係式や真偽値（項の中に関係記号が残っていた場合）は項として扱わない
        candidates = []
    return [expr.subs(_PI_SYMBOL, sympy.pi) for expr in candidates]


def _decimal_tolerance(term: str) -> float:
    """項が小数 1 つなら、表記の最小桁の 1 単位（切り捨て・四捨五入の誤差の上限）を返す。"""
    match = _DECIMAL_TERM.fullmatch(term)
    return 10.0 ** -len(match.group(1)) if match else 0.0


def _evaluate_variants(
    lhs: sympy.Expr, rhs: sympy.Expr, comparison: str, tolerance: float
) -> list[LinkStatus]:
    """そのままの式と、文字 i を虚数単位とみなした式を評価する。評価できたものだけを返す。"""
    variants = [(lhs, rhs)]
    if _I_SYMBOL in lhs.free_symbols | rhs.free_symbols:
        variants.append((lhs.subs(_I_SYMBOL, sympy.I), rhs.subs(_I_SYMBOL, sympy.I)))
    outcomes = [_evaluate(a, b, comparison, tolerance) for a, b in variants]
    return [outcome for outcome in outcomes if outcome is not None]


def _evaluate(
    lhs: sympy.Expr, rhs: sympy.Expr, comparison: str, tolerance: float
) -> LinkStatus | None:
    """標本値を代入して比較する。どの標本値でも数値計算できなければ None。"""
    symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=str)
    integer_symbols = _range_symbols(lhs) | _range_symbols(rhs)
    verdicts: list[bool] = []
    for values in _samples(symbols, integer_symbols):
        left, right = _numeric(lhs, values), _numeric(rhs, values)
        if left is None or right is None:
            continue
        verdict = _compare(left, right, comparison, tolerance)
        if verdict is not None:
            verdicts.append(verdict)
    if not verdicts:
        return None
    if all(verdicts):
        return LinkStatus.HOLDS
    if not any(verdicts):
        return LinkStatus.VIOLATED
    return LinkStatus.CONDITIONAL


def _range_symbols(expr: sympy.Expr) -> set[sympy.Symbol]:
    """総和・総乗の範囲（上限・下限）に現れる文字を返す。"""
    symbols: set[sympy.Symbol] = set()
    for node in expr.atoms(sympy.Sum, sympy.Product):
        for _variable, *bounds in node.limits:
            for bound in bounds:
                symbols |= bound.free_symbols
    return symbols


def _samples(
    symbols: list[sympy.Symbol], integer_symbols: set[sympy.Symbol]
) -> list[dict[sympy.Symbol, float]]:
    """標本値の組を返す。総和・総乗の範囲の文字には整数値（float 型の整数）を入れる。文字が無ければ空の代入 1 組だけを返す。

    文字ごとに候補値の並びを固定シードで並べ替え、どの文字も各候補値を 1 回ずつ取るようにする
    （文字 1 つの条件 a \\le -3 でも、条件を満たす値が必ず試される）。
    """
    if not symbols:
        return [{}]
    rng = random.Random(_SAMPLE_SEED)
    columns: dict[sympy.Symbol, list[float]] = {}
    for symbol in symbols:
        if symbol in integer_symbols:
            column = [float(v) for v in _INTEGER_SAMPLES]  # _numeric で sympy の整数に直す
            column = column * (len(_SAMPLES) // len(column))
        else:
            # 候補値にわずかな揺らぎを加え、別の文字に同じ値が入って偶然等しくなることを避ける
            column = [v * (1 + rng.random() * 0.01) for v in _SAMPLES]
        rng.shuffle(column)
        columns[symbol] = column
    return [{symbol: columns[symbol][i] for symbol in symbols} for i in range(len(_SAMPLES))]


def _numeric(expr: sympy.Expr, values: dict[sympy.Symbol, float]) -> complex | None:
    """式に値を代入して複素数として評価する。評価できない・有限でない場合は None。"""
    # 整数値は sympy の整数として代入する（総和の上限が Float だと evalf の結果が不正確になるため）
    exact = {
        symbol: sympy.Integer(int(v)) if float(v).is_integer() else v
        for symbol, v in values.items()
    }
    try:
        value = complex(expr.evalf(subs=exact))
    except (TypeError, ValueError, ArithmeticError, AttributeError):
        return None
    return value if cmath.isfinite(value) else None


def _compare(left: complex, right: complex, comparison: str, tolerance: float) -> bool | None:
    """2 つの値を比較する。大小比較で値が実数でない場合は None。"""
    margin = _RELATIVE_TOLERANCE * max(1.0, abs(left), abs(right))
    # 小数の近似値の許容誤差は等式だけに使う（不等式の小数は範囲の端として書かれるため）
    if comparison == "=":
        return abs(left - right) <= margin + tolerance
    if comparison == "!=":
        return abs(left - right) > margin + tolerance
    if abs(left.imag) > margin or abs(right.imag) > margin:
        return None
    x, y = left.real, right.real
    if comparison == "<":
        return x < y - margin
    if comparison == ">":
        return x > y + margin
    if comparison == "<=":
        return x <= y + margin
    if comparison == ">=":
        return x >= y - margin
    raise ValueError(f"未対応の比較です: {comparison}")
