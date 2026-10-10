"""仕様: 式番号・例/練習/図などのラベル番号・脚注番号の連番検査で、欠落や誤読の候補を検出する（issue #21）。

教科書の番号は章ごとに 1 から連番で付くため、定義箇所の番号の並びを調べれば、OCR が落とした
式やラベル、誤読した番号を画像なしで見つけられる。参照（「(2.9) より」）を定義と数えると、
定義が欠落しても参照に隠れて見逃すため、両者を区別して抽出する。

定義と参照の区別:
  - 式番号: `\\tag{N.M}`、または「(N.M)」の直後が行末・`$$`・`\\\\`・`\\end{...}` のもの → 定義。
    地の文の途中にある「(N.M)」 → 参照
  - ラベル（例・練習・コメント・図・定理・命題・系・補題・定義）: 太字 `**例 N.M**`、
    または行頭に単独で置かれた「図 N.M」 → 定義。それ以外の言及（「図 1.1 より」） → 参照
  - 脚注: 行頭の `$*N$` / `*N` → 定義。上付きの `^{*N}` / `<sup>\\*N</sup>` → 参照

検査（章 × 種別ごと。式・ラベルの章は番号 N、脚注の章は直近の章見出しから決める）:
  - missing: 1 から最大の定義番号までの欠番。前後の定義があるページと、その番号への参照元を添える
  - duplicate: 同じ番号を複数回定義している
  - order: ページ順に見て、それまでの最大番号より小さい番号が後から現れる
  - undefined_reference: 定義がなく、かつ最大の定義番号を超えている番号への参照

入力の決定:
  - 入力は後処理（normalize_ocr_markdown）適用後の Markdown を既定とする。CLI は
    evaluation.dataset.read_prediction を使うので、ページキャッシュの raw.md には後処理が掛かる。
    後処理は `\\tag{X}` だけの数式を「(X)」に直すが、このモジュールは `\\tag{N.M}` と行末の「(N.M)」
    のどちらも定義とみなすため、後処理の前後どちらの Markdown でも同じ結果になる
  - 後処理が見出しに変えたラベル（`### 例 2.1`）は太字化後の形で扱うので、後処理前の見出し形式も
    行頭の単独ラベルとして定義とみなす

注意事項:
  - 章見出し（`# N 題` / `## N.M 題`）より前の脚注は 0 章として扱う。目次の項目のようにページ番号で
    終わる見出しは章見出しとみなさない
  - 「(0.5)」のような小数は、章番号が 1〜MAX_CHAPTER の範囲外なので式番号とみなさない
  - コードブロックの中は、行頭に単独で置かれたラベル（文字で描いた図の `図 11.1` など）だけを定義として拾い、
    それ以外は無視する
  - 並行する検出器との競合を避けるため、detectors/cli.py には組み込まず単独で実行する:
    `uv run python -m ouj_notebook_converter.detectors.numbering --pred <ディレクトリ> [--json 結果.json]`
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.evaluation.dataset import list_prediction_pages, read_prediction
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import split_code_fences

# 章番号として認める上限。これを超える「(N.M)」は小数などとみなす
MAX_CHAPTER = 20

EQUATION = "式"
FOOTNOTE = "脚注"
LABEL_KINDS = ("例", "練習", "コメント", "図", "定理", "命題", "系", "補題", "定義")
# 表示順
KIND_ORDER = (EQUATION, *LABEL_KINDS, FOOTNOTE)

FINDING_MISSING = "missing"
FINDING_DUPLICATE = "duplicate"
FINDING_ORDER = "order"
FINDING_UNDEFINED_REFERENCE = "undefined_reference"

_HEADING = re.compile(r"^#{1,2}\s+(\d{1,2})(?:\.\d{1,3})?\s+\S.*$")
# 目次の項目のように、行末がページ番号になっている見出し
_ENDS_WITH_PAGE_NUMBER = re.compile(r"\s\d+\s*$")
_TAG = re.compile(r"\\tag\{(\d{1,2})\.(\d{1,3})\}")
_PAREN_NUMBER = re.compile(r"\((\d{1,2})\.(\d{1,3})\)")
# 「(N.M)」の直後がこれなら、行末に置かれた式番号（定義）とみなす
_EQUATION_NUMBER_TAIL = re.compile(r"^\s*(?:$|\$(?!\s*_)|\\\\|\\end\{|&)")
_LABEL_ALTERNATIVES = "|".join(LABEL_KINDS)
# 漢字・かな・英数字に続く「系」「例」（「関係 1.5」など）をラベルとみなさない
_LABEL = re.compile(
    rf"(?<![\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9])(?P<bold>\*\*)?(?P<kind>{_LABEL_ALTERNATIVES})"
    r"\s*(?P<chapter>\d{1,2})\.(?P<number>\d{1,3})"
)
# `<div align="center">図 2.7</div>` のようにラベルを包む HTML タグ
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_LABEL_ALONE_ON_LINE = re.compile(r"^(?:#{1,6}\s+)?\s*\**\s*$")
# 行頭の脚注記号。`$*N$` / `*N` のほか、上付き形式 `$^{*N}$` で書かれた脚注本文も定義とみなす
_FOOTNOTE_DEFINITION = re.compile(
    r"^\s*\$?\s*(?:(?:\{\})?\^\{?)?\\?\*(\d{1,3})(?=$|[\s$\\}\u3000])"
)
_FOOTNOTE_REFERENCE = re.compile(r"(?:\^|<sup>)\s*\{?\s*\\?\*(\d{1,3})")


@dataclass(frozen=True)
class Occurrence:
    """番号の 1 回の出現。line は 1 始まりのページ内の行番号。"""

    kind: str
    chapter: int
    number: int
    page: int
    line: int
    is_definition: bool


@dataclass
class Finding:
    """連番検査の指摘 1 件。

    type ごとの使い方:
      missing: after_page / before_page に前後の定義があるページ、references に参照元
      duplicate: definitions に全ての定義
      order: definitions に順序が逆転した定義
      undefined_reference: references に全ての参照
    """

    type: str
    kind: str
    chapter: int
    number: int
    definitions: list[Occurrence] = field(default_factory=list)
    references: list[Occurrence] = field(default_factory=list)
    after_page: int | None = None
    before_page: int | None = None


@dataclass
class NumberingReport:
    """連番検査の結果。"""

    page_count: int
    definition_counts: dict[str, int]
    reference_counts: dict[str, int]
    findings: list[Finding]

    def finding_counts(self) -> dict[str, int]:
        """指摘の種類（type）ごとの件数を返す。"""
        return dict(Counter(f.type for f in self.findings))

    def to_dict(self) -> dict[str, Any]:
        """JSON に書き出せる辞書にする。"""
        return {
            "page_count": self.page_count,
            "definition_counts": self.definition_counts,
            "reference_counts": self.reference_counts,
            "finding_counts": self.finding_counts(),
            "findings": [asdict(f) for f in self.findings],
        }


def _heading_chapter(line: str) -> int | None:
    """章見出し（`# N 題` / `## N.M 題`）なら章番号を返す。目次の項目などは None。"""
    match = _HEADING.match(line)
    if match is None or _ENDS_WITH_PAGE_NUMBER.search(line):
        return None
    chapter = int(match.group(1))
    return chapter if 1 <= chapter <= MAX_CHAPTER else None


def _scan_labels(page: int, line_no: int, line: str) -> list[tuple[int, Occurrence]]:
    """1 行からラベル（例・図など）の出現を (位置, 出現) の組で抽出する。"""
    found: list[tuple[int, Occurrence]] = []
    for m in _LABEL.finditer(line):
        ch = int(m.group("chapter"))
        if not 1 <= ch <= MAX_CHAPTER:
            continue
        rest_of_line = _HTML_TAG.sub("", line[: m.start()] + line[m.end() :])
        alone = _LABEL_ALONE_ON_LINE.match(rest_of_line) is not None
        is_definition = bool(m.group("bold")) or alone
        occurrence = Occurrence(
            m.group("kind"), ch, int(m.group("number")), page, line_no, is_definition
        )
        found.append((m.start(), occurrence))
    return found


def _scan_line(page: int, line_no: int, line: str, chapter: int) -> list[Occurrence]:
    """1 行から番号の出現を抽出する（位置の昇順）。chapter は脚注の章。"""
    found: list[tuple[int, Occurrence]] = []

    def add(position: int, kind: str, ch: int, number: int, is_definition: bool) -> None:
        found.append((position, Occurrence(kind, ch, number, page, line_no, is_definition)))

    footnote_def = _FOOTNOTE_DEFINITION.match(line)
    if footnote_def is not None:
        add(footnote_def.start(1), FOOTNOTE, chapter, int(footnote_def.group(1)), True)
    for m in _FOOTNOTE_REFERENCE.finditer(line):
        if footnote_def is not None and m.start() < footnote_def.end():
            continue  # 行頭の `$^{*N}$` は定義の記号であり、参照ではない
        add(m.start(), FOOTNOTE, chapter, int(m.group(1)), False)
    for m in _TAG.finditer(line):
        add(m.start(), EQUATION, int(m.group(1)), int(m.group(2)), True)
    for m in _PAREN_NUMBER.finditer(line):
        ch = int(m.group(1))
        if 1 <= ch <= MAX_CHAPTER:
            is_definition = _EQUATION_NUMBER_TAIL.match(line[m.end() :]) is not None
            add(m.start(), EQUATION, ch, int(m.group(2)), is_definition)
    found.extend(_scan_labels(page, line_no, line))
    found.sort(key=lambda item: item[0])
    return [occurrence for _, occurrence in found]


def extract_occurrences(
    page: int, markdown: str, chapter: int | None
) -> tuple[list[Occurrence], int | None]:
    """1 ページ分の Markdown から番号の定義・参照を抽出する。

    Args:
        page: ページ番号（1 始まり）。
        markdown: ページの Markdown。
        chapter: このページの先頭時点の章。脚注の章に使う。None は章見出しより前（0 章）。

    Returns:
        (番号の出現のリスト, ページ末尾時点の章)。
    """
    occurrences: list[Occurrence] = []
    line_no = 0
    for segment, in_fence in split_code_fences(markdown):
        for line in segment.splitlines():
            line_no += 1
            if in_fence:
                # 図を文字で描いたコードブロックには、図番号の行だけが残ることがある。
                # 式番号・脚注は誤抽出を避けるため、ラベルの定義だけを拾う
                occurrences.extend(
                    o for _, o in _scan_labels(page, line_no, line) if o.is_definition
                )
                continue
            if line.lstrip().startswith("#"):
                heading_chapter = _heading_chapter(line)
                if heading_chapter is not None:
                    chapter = heading_chapter
                    continue
            occurrences.extend(_scan_line(page, line_no, line, chapter or 0))
    return occurrences, chapter


def _kind_rank(kind: str) -> int:
    return KIND_ORDER.index(kind)


def _check_group(
    kind: str, chapter: int, definitions: list[Occurrence], references: list[Occurrence]
) -> list[Finding]:
    """同じ種別・章の定義と参照を検査する。definitions は文書順。"""
    findings: list[Finding] = []
    by_number: dict[int, list[Occurrence]] = {}
    for d in definitions:
        by_number.setdefault(d.number, []).append(d)
    refs_by_number: dict[int, list[Occurrence]] = {}
    for r in references:
        refs_by_number.setdefault(r.number, []).append(r)

    for number, defs in sorted(by_number.items()):
        if len(defs) > 1:
            findings.append(Finding(FINDING_DUPLICATE, kind, chapter, number, definitions=defs))

    seen: set[int] = set()
    running_max = 0
    for d in definitions:
        if d.number < running_max and d.number not in seen:
            findings.append(Finding(FINDING_ORDER, kind, chapter, d.number, definitions=[d]))
        seen.add(d.number)
        running_max = max(running_max, d.number)

    defined_max = max(by_number, default=0)
    for number in range(1, defined_max + 1):
        if number in by_number:
            continue
        lower = [n for n in by_number if n < number]
        upper = [n for n in by_number if n > number]
        findings.append(
            Finding(
                FINDING_MISSING,
                kind,
                chapter,
                number,
                references=refs_by_number.get(number, []),
                after_page=by_number[max(lower)][0].page if lower else None,
                before_page=by_number[min(upper)][0].page if upper else None,
            )
        )
    for number, refs in sorted(refs_by_number.items()):
        if number > defined_max:
            findings.append(
                Finding(FINDING_UNDEFINED_REFERENCE, kind, chapter, number, references=refs)
            )
    return findings


def check_numbering(pages: Iterable[tuple[int, str]]) -> NumberingReport:
    """全ページの Markdown から番号を抽出し、連番を検査する。

    Args:
        pages: (ページ番号, Markdown) の並び。ページ番号の昇順に処理する。

    Returns:
        検査結果。指摘は (種別, 章, 番号) の順に並ぶ。
    """
    occurrences: list[Occurrence] = []
    page_count = 0
    chapter: int | None = None
    for page, markdown in sorted(pages, key=lambda item: item[0]):
        page_count += 1
        found, chapter = extract_occurrences(page, markdown, chapter)
        occurrences.extend(found)
    return check_occurrences(occurrences, page_count)


def check_occurrences(occurrences: list[Occurrence], page_count: int) -> NumberingReport:
    """抽出済みの番号の出現（文書順）から連番を検査する。"""
    groups: dict[tuple[str, int], tuple[list[Occurrence], list[Occurrence]]] = {}
    for o in occurrences:
        definitions, references = groups.setdefault((o.kind, o.chapter), ([], []))
        (definitions if o.is_definition else references).append(o)

    findings: list[Finding] = []
    for (kind, ch), (definitions, references) in sorted(
        groups.items(), key=lambda item: (_kind_rank(item[0][0]), item[0][1])
    ):
        findings.extend(_check_group(kind, ch, definitions, references))
    findings.sort(key=lambda f: (_kind_rank(f.kind), f.chapter, f.number))

    def count(is_definition: bool) -> dict[str, int]:
        counter = Counter(o.kind for o in occurrences if o.is_definition == is_definition)
        return {kind: counter[kind] for kind in KIND_ORDER if counter[kind]}

    return NumberingReport(
        page_count=page_count,
        definition_counts=count(True),
        reference_counts=count(False),
        findings=findings,
    )


app = typer.Typer(add_completion=False)


@app.command()
def numbering(
    pred: Annotated[
        Path, typer.Option("--pred", help="検査対象（--no-combine 出力 または ページキャッシュ）")
    ],
    json_path: Annotated[
        Path | None, typer.Option("--json", help="検査結果を書き出す JSON ファイル")
    ] = None,
) -> None:
    """式番号・ラベル番号・脚注番号の連番を検査し、欠番・重複・順序の逆転を表示する。"""
    try:
        pages = [(page, read_prediction(pred, page)) for page in list_prediction_pages(pred)]
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e
    report = check_numbering(pages)

    console = Console()
    console.print(_summary_table(report))
    console.print(_findings_table(report.findings))
    if json_path is not None:
        try:
            json_path.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(code=1) from e
        console.print(f"検査結果を書き出しました: {json_path}")


def _summary_table(report: NumberingReport) -> Table:
    table = Table(title=f"番号の抽出件数（{report.page_count} ページ）")
    for column in ("種別", "定義", "参照"):
        table.add_column(column)
    for kind in KIND_ORDER:
        defs = report.definition_counts.get(kind, 0)
        refs = report.reference_counts.get(kind, 0)
        if defs or refs:
            table.add_row(kind, str(defs), str(refs))
    return table


def _location(findings: Finding) -> str:
    """指摘の場所を人が読める文字列にする。"""
    if findings.type == FINDING_MISSING:
        after = f"p.{findings.after_page}" if findings.after_page else "章の先頭"
        before = f"p.{findings.before_page}" if findings.before_page else "章の末尾"
        refs = "".join(f" 参照 p.{r.page}:{r.line}" for r in findings.references)
        return f"{after} と {before} の間{refs}"
    occurrences = findings.definitions or findings.references
    return " ".join(f"p.{o.page}:{o.line}" for o in occurrences)


def _findings_table(findings: list[Finding]) -> Table:
    table = Table(title=f"指摘（{len(findings)} 件）")
    for column in ("種類", "種別", "番号", "場所"):
        table.add_column(column)
    for f in findings:
        label = f"*{f.number}" if f.kind == FOOTNOTE else f"{f.chapter}.{f.number}"
        chapter = f" (第{f.chapter}章)" if f.kind == FOOTNOTE else ""
        table.add_row(f.type, f.kind, label + chapter, _location(f))
    return table


if __name__ == "__main__":
    app()
