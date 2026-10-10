"""仕様: 各検出器の誤読候補を JudgeCandidate（共通形式）に正規化する。

対象の検出器:
  - equation_check: 等式の検算で成り立たない式（issue #19）
  - rare_word: 書籍内で低頻度、かつ高頻度語と 1 文字違いの語（issue #20）
  - cross_ocr_diff: Gemini と yomitoku の地の文の食い違い（issue #23。yomitoku の bbox を持つ）
  - katex_error: KaTeX で描画できない数式（issue #22）

連番検査（issue #21）は「無いもの」の報告で、置換の対象となる文字列が無いため判定対象にしない。

検出器には後処理（normalize_ocr_markdown）前の raw.md をそのまま渡す。修正ファイル（#28）の
`before` は raw.md 上の表記で書くため、LLM に見せる抜粋も raw.md の行にそろえる。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from ouj_notebook_converter.detectors.consistency import (
    Candidate as RareWordCandidate,
)
from ouj_notebook_converter.detectors.consistency import (
    detect_index_pages,
    find_rare_word_candidates,
)
from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    MATH_LABEL,
    MATH_MARKER,
    _Analysis,
    compare_page,
    extract_gemini_prose,
    parse_analysis,
)
from ouj_notebook_converter.detectors.cross_ocr_diff.detect import (
    Candidate as DiffCandidate,
)
from ouj_notebook_converter.detectors.equation_check import LinkStatus, check_markdown
from ouj_notebook_converter.evaluation.dataset import list_prediction_pages
from ouj_notebook_converter.evaluation.katex import check_katex
from ouj_notebook_converter.judge.locate import align_lines
from ouj_notebook_converter.judge.models import ALL_SOURCES, JudgeCandidate, Source
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import normalize_ocr_markdown
from ouj_notebook_converter.pipeline.types import PageMarkdown
from ouj_notebook_converter.quality_report import build_katex_report

_PAGE_FILE = re.compile(r"page_(\d+)")
# 抜粋に含める、該当行の前後の文脈（非空行）の数
_NEIGHBOR_LINES = 1
# 抜粋の位置合わせに使う文脈の長さ（長い順に試す）
_CONTEXT_LENGTHS = (15, 8, 4, 0)


@dataclass(frozen=True)
class Excerpt:
    """raw.md の抜粋。text は該当行と前後の非空行、line_index は該当行（先頭）の 0 始まりの行番号。"""

    text: str
    line_index: int


def read_raw_pages(pred_dir: Path) -> dict[int, str]:
    """ページ番号（1 始まり）→ 後処理前の Markdown を読む。

    ページキャッシュ（page_NNNN/raw.md）と --no-combine 出力（page_NNNN.md）の両方に対応する。

    Raises:
        FileNotFoundError: ディレクトリが無い、またはページが 1 つも無い場合。
        ValueError: 同じページに両方の構成のファイルがある場合。
    """
    pages: dict[int, str] = {}
    for page in list_prediction_pages(pred_dir):
        output_path = pred_dir / f"page_{page:04d}.md"
        cache_path = pred_dir / f"page_{page:04d}" / "raw.md"
        if output_path.is_file() and cache_path.is_file():
            raise ValueError(f"{output_path} と {cache_path} の両方が存在します")
        pages[page] = (output_path if output_path.is_file() else cache_path).read_text(
            encoding="utf-8"
        )
    if not pages:
        raise FileNotFoundError(f"ページが見つかりません: {pred_dir}")
    return pages


def _excerpt_of_lines(lines: list[str], start: int, end: int) -> Excerpt:
    """lines[start..end] と、その前後の非空行 _NEIGHBOR_LINES 行ずつを抜粋にする。"""
    first = start
    for _ in range(_NEIGHBOR_LINES):
        first -= 1
        while first > 0 and not lines[first].strip():
            first -= 1
    first = max(first, 0)
    last = end
    for _ in range(_NEIGHBOR_LINES):
        last += 1
        while last < len(lines) - 1 and not lines[last].strip():
            last += 1
    last = min(last, len(lines) - 1)
    return Excerpt(text="\n".join(lines[first : last + 1]), line_index=start)


def find_excerpt(raw: str, needle: str) -> Excerpt | None:
    """raw の中で needle が最初に現れる行を中心に抜粋を作る。見つからなければ None。"""
    if not needle:
        return None
    index = raw.find(needle)
    if index < 0:
        return None
    start = raw.count("\n", 0, index)
    return _excerpt_of_lines(raw.split("\n"), start, start + needle.count("\n"))


def _make(
    page: int,
    source: Source,
    kind: str,
    raw: str,
    needle: str,
    *,
    text: str,
    hint: str,
    **extra: object,
) -> JudgeCandidate:
    """抜粋を探して JudgeCandidate を作る。抜粋が無ければページ全体を渡す。"""
    excerpt = find_excerpt(raw, needle)
    return JudgeCandidate(
        id="",
        page=page,
        source=source,
        kind=kind,
        text=text,
        hint=hint,
        excerpt=excerpt.text if excerpt is not None else raw,
        excerpt_scope="lines" if excerpt is not None else "page",
        line_index=excerpt.line_index if excerpt is not None else None,
        **extra,  # type: ignore[arg-type]
    )


def equation_candidates(raw_pages: Mapping[int, str]) -> list[JudgeCandidate]:
    """成り立たない等式・不等式を含む数式を候補にする（数式ごとに 1 件）。"""
    candidates: list[JudgeCandidate] = []
    for page, raw in sorted(raw_pages.items()):
        seen: set[int] = set()
        for result in check_markdown(raw).links:
            if result.status != LinkStatus.VIOLATED or result.formula_index in seen:
                continue
            seen.add(result.formula_index)
            link = result.link
            candidates.append(
                _make(
                    page,
                    "equation_check",
                    "violated",
                    raw,
                    result.formula,
                    text=result.formula,
                    hint=(
                        f"記号計算で「{link.lhs}」と「{link.rhs}」の関係 {link.relation} が"
                        "どんな値でも成り立ちませんでした"
                    ),
                )
            )
    return candidates


def rare_word_candidates(
    raw_pages: Mapping[int, str], *, skip_index: bool = False
) -> list[JudgeCandidate]:
    """書籍内で低頻度、かつ高頻度語と 1 文字違いの語を候補にする。

    Raises:
        ValueError: 索引ページを検出できず、skip_index も指定されていない場合。
    """
    index_pages = detect_index_pages(raw_pages)
    if not index_pages and not skip_index:
        raise ValueError(
            "索引ページを検出できませんでした（索引の無い本は skip_index を指定してください）"
        )
    body_pages = {p: md for p, md in raw_pages.items() if p not in index_pages}
    found: list[RareWordCandidate] = find_rare_word_candidates(body_pages)
    return [
        _make(
            c.page,
            "rare_word",
            "rare_word",
            raw_pages[c.page],
            c.surface,
            text=c.surface,
            hint=(
                f"この本で「{c.surface}」は {c.count} 回しか出ませんが、1 文字違いの"
                f"「{c.suggestion}」は {c.suggestion_count} 回出ます"
            ),
        )
        for c in sorted(found, key=lambda c: (c.page, c.start))
    ]


def _locate_diff_line(
    raw: str, candidate: DiffCandidate, analysis: _Analysis | None
) -> Excerpt | None:
    """cross_ocr_diff の候補の Gemini 側の文脈が現れる行を探し、抜粋を作る。

    候補の文脈は地の文の正規化形（Markdown 記法・空白なし）なので、行ごとに同じ正規化をして照合する。
    文脈が行をまたぐ場合に備え、長い文脈から短い文脈の順に試し、最初に一致があった長さで決める。
    一致する行が複数ある場合は、先頭の行を選ぶと別の行を LLM に見せて誤った修正を出させるため、
    候補の bbox の中心を含む yomitoku の段落に対応付けられた行だけに絞る。1 行に決まらなければ None。
    """
    lines = raw.split("\n")
    prose_lines = [extract_gemini_prose(line).replace(MATH_MARKER, MATH_LABEL) for line in lines]
    for length in _CONTEXT_LENGTHS:
        needle = (
            (candidate.gemini_before[-length:] if length else "")
            + candidate.gemini_text
            + (candidate.gemini_after[:length] if length else "")
        )
        if not needle:
            continue
        matches = [i for i, prose in enumerate(prose_lines) if needle in prose]
        if not matches:
            continue
        if len(matches) > 1 and analysis is not None:
            aligned = align_lines(lines, analysis)
            center_x = (candidate.bbox[0] + candidate.bbox[2]) / 2
            center_y = (candidate.bbox[1] + candidate.bbox[3]) / 2
            matches = [
                i
                for i in matches
                if i in aligned
                and aligned[i][0] <= center_x <= aligned[i][2]
                and aligned[i][1] <= center_y <= aligned[i][3]
            ]
        if len(matches) != 1:
            return None
        return _excerpt_of_lines(lines, matches[0], matches[0])
    return None


def _diff_hint(c: DiffCandidate) -> str:
    if c.kind == "replace":
        return f"別の OCR（yomitoku）はこの箇所を「{c.yomitoku_text}」と読んでいます"
    if c.kind == "gemini_only":
        return "別の OCR（yomitoku）はこの箇所に文字を読んでいません（Gemini の読み過ぎの疑い）"
    return f"別の OCR（yomitoku）は「{c.yomitoku_text}」を読んでいますが、Gemini には無い（脱落の疑い）"


def cross_ocr_candidates(raw_pages: Mapping[int, str], yomitoku_dir: Path) -> list[JudgeCandidate]:
    """Gemini と yomitoku の地の文の食い違いを候補にする。位置は yomitoku の bbox を使う。

    Raises:
        FileNotFoundError: yomitoku の analysis.json が無いページがある場合。
        ValueError: analysis.json の形式が不正な場合。
    """
    candidates: list[JudgeCandidate] = []
    for page, raw in sorted(raw_pages.items()):
        analysis_path = yomitoku_dir / f"page_{page:04d}" / "analysis.json"
        if not analysis_path.is_file():
            raise FileNotFoundError(f"yomitoku の analysis.json が見つかりません: {analysis_path}")
        analysis = parse_analysis(
            json.loads(analysis_path.read_text(encoding="utf-8")), source=str(analysis_path)
        )
        for c in compare_page(page, raw, analysis).candidates:
            excerpt = _locate_diff_line(raw, c, analysis)
            candidates.append(
                JudgeCandidate(
                    id="",
                    page=page,
                    source="cross_ocr_diff",
                    kind=c.kind,
                    text=c.gemini_text,
                    hint=_diff_hint(c),
                    excerpt=excerpt.text if excerpt is not None else raw,
                    excerpt_scope="lines" if excerpt is not None else "page",
                    line_index=excerpt.line_index if excerpt is not None else None,
                    context_before=c.gemini_before,
                    context_after=c.gemini_after,
                    yomitoku_text=c.yomitoku_text,
                    bbox=c.bbox,
                    location="yomitoku_word" if c.bbox_source == "word" else "yomitoku_paragraph",
                )
            )
    return candidates


def katex_candidates(raw_pages: Mapping[int, str]) -> list[JudgeCandidate]:
    """KaTeX で描画できない数式を候補にする（変換時の後処理を適用した状態で検査する）。

    Raises:
        KatexCheckError: Node.js や katex が使えない場合。
    """
    pages = [
        PageMarkdown(page_index=page - 1, markdown=normalize_ocr_markdown(raw))
        for page, raw in sorted(raw_pages.items())
    ]
    report = build_katex_report(pages, check=check_katex)
    return [
        _make(
            error.page,
            "katex_error",
            "render_error",
            raw_pages[error.page],
            error.tex,
            text=error.tex,
            hint=f"KaTeX で描画できません: {error.message}",
        )
        for error in report.errors
    ]


def collect_candidates(
    raw_pages: Mapping[int, str],
    *,
    yomitoku_dir: Path | None,
    sources: Collection[str] = ALL_SOURCES,
    skip_index: bool = False,
) -> list[JudgeCandidate]:
    """指定した検出器の候補を集め、ページ順に `pNNNN-<source>-<n>` の id を振って返す。

    Raises:
        ValueError: cross_ocr_diff を要求したのに yomitoku_dir が無い場合、未知の検出器名の場合。
    """
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise ValueError(f"未知の検出器です: {sorted(unknown)}")
    found: list[JudgeCandidate] = []
    if "equation_check" in sources:
        found += equation_candidates(raw_pages)
    if "rare_word" in sources:
        found += rare_word_candidates(raw_pages, skip_index=skip_index)
    if "cross_ocr_diff" in sources:
        if yomitoku_dir is None:
            raise ValueError("cross_ocr_diff には yomitoku のキャッシュ（yomitoku_dir）が必要です")
        found += cross_ocr_candidates(raw_pages, yomitoku_dir)
    if "katex_error" in sources:
        found += katex_candidates(raw_pages)

    order = {source: i for i, source in enumerate(ALL_SOURCES)}
    found.sort(key=lambda c: (c.page, order[c.source]))
    counters: Counter[tuple[int, str]] = Counter()
    result: list[JudgeCandidate] = []
    for c in found:
        counters[(c.page, c.source)] += 1
        n = counters[(c.page, c.source)]
        result.append(JudgeCandidate(**{**c.__dict__, "id": f"p{c.page:04d}-{c.source}-{n}"}))
    return result
