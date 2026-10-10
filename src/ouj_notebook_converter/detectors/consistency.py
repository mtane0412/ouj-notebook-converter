"""仕様: 書籍内の表記の一貫性から、地の文の OCR 誤字候補を検出する（issue #20 のプロトタイプ）。

着想: 奥付の著者名「服部正博」は本全体で 1 回しか出ない語で、正しい「隈部正博」は何度も出る。
このように「めったに出ない語が、よく出る語と 1 文字だけ違う」場合は、誤字の可能性が高い。

3 つの検出器を持つ（Candidate.kind で区別する）:
  - rare_word:  本全体で max_rare_count 回以下しか出ない語のうち、編集距離 1 で出現回数が
                min_neighbor_count 回以上の語がある語を候補にする
  - index_near: 巻末索引の語彙を辞書とし、索引に無く出現の少ない本文中の語のうち、
                編集距離 1 の索引語がある語を候補にする
  - rare_ngram: 分かち書きできないひらがな（「としとう」など）向けに、かな 1 文字違いで
                出現の少ない文字 n-gram を候補にする（重なる n-gram は 1 件にまとめる）

日本語の分かち書きは依存を増やさず、文字種の連続で代用する:
  - 語 = 漢字の連続、またはカタカナの連続（2 文字以上）。ひらがなは語に含めない
  - ひらがなの誤字は rare_ngram で拾う

評価（evaluate_page）: 評価セットの正解との文字差分（difflib）を「真の誤読」とみなす。
  - 誤読箇所 = 評価対象の地の文（body_text）と正解の地の文の差分のうち、一致しない区間
  - 候補が的中 = 候補の文字範囲が誤読箇所のどれかと重なる。適合率 = 的中した候補 / 候補
  - 誤読箇所が検出 = いずれかの候補と重なる。再現率 = 検出された誤読箇所 / 誤読箇所
  - 数式・記号・空白・英数字の違いは比較しない（地の文の誤字だけを対象にするため）

注意事項:
  - 語頭・語末に 1 文字足しただけの語（整数と整数列）は複合語・派生語とみなして近い語に数えない
  - 候補の位置（start, end）は、そのページの body_text 上の文字範囲
  - 索引ページは本文の語の数え上げと候補の対象から除く（索引語は 1 回ずつしか出ず、頻度の偏りが無いため）。
    除外は呼び出し側（CLI）が detect_index_pages で行う
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.evaluation.dataset import (
    list_prediction_pages,
    load_manifest,
    read_prediction,
    read_truth,
)
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN

CandidateKind = Literal["rare_word", "index_near", "rare_ngram"]

# 地の文として残す文字: ひらがな・カタカナ（長音符を含む）・漢字（々を含む）
_NOT_KEPT = re.compile(r"[^ぁ-ゟァ-ー一-鿿々]+")
# 語: 漢字の連続、またはカタカナの連続（いずれも 2 文字以上）
_WORD = re.compile(r"[一-鿿々]{2,}|[ァ-ー]{2,}")
_HIRAGANA = re.compile(r"[ぁ-ゟ]")
_SEGMENT = re.compile(r"[^ ]+")
_SEPARATOR = " "

# 索引の行: 「用語　89」「用語　241, 254」の形（用語は数字・記号で始めない）
_INDEX_LINE = re.compile(r"^(?P<term>[^\s$#●][^$]*?)[\s　]+\d+(?:\s*[,、，]\s*\d+)*\s*$")
_INDEX_MIN_LINES = 10
_INDEX_MIN_RATIO = 0.6
_INDEX_TERM_MIN_LENGTH = 2

DEFAULT_MAX_RARE_COUNT = 2
DEFAULT_MIN_NEIGHBOR_COUNT = 3
DEFAULT_MIN_WORD_LENGTH = 3
DEFAULT_INDEX_MAX_COUNT = 3
DEFAULT_NGRAM_SIZE = 4
DEFAULT_NGRAM_MAX_RARE_COUNT = 1
DEFAULT_NGRAM_MIN_NEIGHBOR_COUNT = 10


@dataclass(frozen=True)
class Candidate:
    """誤字候補 1 件。

    Attributes:
        page: ページ番号（1 始まり）。
        kind: 検出器の種類。
        surface: 本文中の表記（誤字の疑いがある文字列）。
        suggestion: 近い（正しい可能性のある）表記。
        count: 本全体での surface の出現回数。
        suggestion_count: 本全体での suggestion の出現回数（索引語のみで本文に無ければ 0）。
        start: ページの body_text 上の開始位置。
        end: ページの body_text 上の終了位置（含まない）。
    """

    page: int
    kind: CandidateKind
    surface: str
    suggestion: str
    count: int
    suggestion_count: int
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        """JSON に書き出すための辞書を返す。"""
        return {
            "page": self.page,
            "kind": self.kind,
            "surface": self.surface,
            "suggestion": self.suggestion,
            "count": self.count,
            "suggestion_count": self.suggestion_count,
        }


@dataclass(frozen=True)
class PageScore:
    """誤字候補と真の誤読箇所の件数。

    Attributes:
        flagged: 誤字候補の件数。
        true_positive: 誤読箇所と重なった候補の件数。
        misread: 誤読箇所（正解との差分の区間）の件数。
        detected: いずれかの候補と重なった誤読箇所の件数。
    """

    flagged: int
    true_positive: int
    misread: int
    detected: int

    @property
    def precision(self) -> float:
        """適合率。候補が 0 件なら誤検出も無いため 1 とする。"""
        return self.true_positive / self.flagged if self.flagged else 1.0

    @property
    def recall(self) -> float:
        """再現率。誤読箇所が 0 件なら見落としも無いため 1 とする。"""
        return self.detected / self.misread if self.misread else 1.0

    def to_dict(self) -> dict[str, int | float]:
        """JSON に書き出すための辞書を返す。"""
        return {
            "flagged": self.flagged,
            "true_positive": self.true_positive,
            "misread": self.misread,
            "detected": self.detected,
            "precision": self.precision,
            "recall": self.recall,
        }


def sum_page_scores(scores: Iterable[PageScore]) -> PageScore:
    """複数ページの件数を合計する。"""
    items = list(scores)
    return PageScore(
        flagged=sum(s.flagged for s in items),
        true_positive=sum(s.true_positive for s in items),
        misread=sum(s.misread for s in items),
        detected=sum(s.detected for s in items),
    )


def body_text(markdown: str) -> str:
    """1 ページ分の Markdown から、日本語の文字だけの地の文を作る。

    数式・Markdown 記法・句読点・英数字は除き、残った文字の連なりを半角空白 1 つで区切る。
    """
    without_math = MATH_SPAN.sub(_SEPARATOR, markdown)
    return _NOT_KEPT.sub(_SEPARATOR, without_math).strip(_SEPARATOR)


def detect_index_pages(pages: Mapping[int, str]) -> set[int]:
    """巻末索引のページを返す。

    「用語　ページ番号」の形の行が大半を占めるページのうち、最後尾の連続したページだけを索引とする
    （目次も同じ形の行が並ぶため、巻頭の目次を索引に含めないように）。
    """
    matched_pages: set[int] = set()
    for page, markdown in pages.items():
        lines = [
            line.strip()
            for line in markdown.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        matched = sum(1 for line in lines if _INDEX_LINE.match(line))
        if matched >= _INDEX_MIN_LINES and matched / len(lines) >= _INDEX_MIN_RATIO:
            matched_pages.add(page)
    index_pages: set[int] = set()
    page = max(matched_pages, default=0)
    while page in matched_pages:
        index_pages.add(page)
        page -= 1
    return index_pages


def parse_index_terms(markdown: str) -> set[str]:
    """索引ページの Markdown から、漢字・カタカナだけの索引語を取り出す。

    数式を含む行・ひらがなを含む語は、語の単位で比較できないため除く。
    """
    terms: set[str] = set()
    for line in markdown.splitlines():
        match = _INDEX_LINE.match(line.strip())
        if match is None:
            continue
        term = match.group("term").strip()
        if len(term) >= _INDEX_TERM_MIN_LENGTH and _WORD.fullmatch(term):
            terms.add(term)
    return terms


def find_rare_word_candidates(
    pages: Mapping[int, str],
    *,
    max_rare_count: int = DEFAULT_MAX_RARE_COUNT,
    min_neighbor_count: int = DEFAULT_MIN_NEIGHBOR_COUNT,
    min_length: int = DEFAULT_MIN_WORD_LENGTH,
) -> list[Candidate]:
    """低頻度語のうち、編集距離 1 の高頻度語がある語を候補にする。

    Args:
        pages: ページ番号 → 本文の Markdown（索引ページは含めない）。
        max_rare_count: 低頻度とみなす出現回数の上限。
        min_neighbor_count: 高頻度とみなす近い語の出現回数の下限。
        min_length: 候補にする語の最小文字数（短い語は別の正しい語と偶然近くなりやすいため）。
    """
    bodies = _bodies(pages)
    counts = _count_words(bodies)
    frequent = {word for word, count in counts.items() if count >= min_neighbor_count}
    return _word_candidates(
        bodies, counts, _NeighborIndex(frequent), "rare_word", max_rare_count, min_length
    )


def find_index_near_candidates(
    pages: Mapping[int, str],
    index_terms: Collection[str],
    *,
    max_count: int = DEFAULT_INDEX_MAX_COUNT,
    min_length: int = DEFAULT_MIN_WORD_LENGTH,
) -> list[Candidate]:
    """索引に無く出現の少ない本文中の語のうち、編集距離 1 の索引語がある語を候補にする。

    Args:
        pages: ページ番号 → 本文の Markdown（索引ページは含めない）。
        index_terms: 索引語（parse_index_terms の結果）。
        max_count: 候補にする語の出現回数の上限。
        min_length: 候補にする語の最小文字数。
    """
    bodies = _bodies(pages)
    counts = _count_words(bodies)
    # 索引に載っている語そのものは正しい表記なので、候補の対象から除く
    return _word_candidates(
        bodies,
        counts,
        _NeighborIndex(index_terms),
        "index_near",
        max_count,
        min_length,
        exclude=index_terms,
    )


def find_rare_ngram_candidates(
    pages: Mapping[int, str],
    *,
    n: int = DEFAULT_NGRAM_SIZE,
    max_rare_count: int = DEFAULT_NGRAM_MAX_RARE_COUNT,
    min_neighbor_count: int = DEFAULT_NGRAM_MIN_NEIGHBOR_COUNT,
) -> list[Candidate]:
    """ひらがな 1 文字だけ違う高頻度の n-gram がある、低頻度の n-gram を候補にする。

    ひらがなの誤字（「しよう」→「しとう」）は語として切り出せないため、n 文字の窓で探す。
    違う 1 文字が両方ひらがなの場合だけ候補にする（漢字は「二次」「三次」のように
    正しい語どうしが 1 文字違いになりやすいため）。重なる n-gram は 1 件にまとめる。

    Args:
        pages: ページ番号 → 本文の Markdown（索引ページは含めない）。
        n: n-gram の文字数。
        max_rare_count: 低頻度とみなす n-gram の出現回数の上限。
        min_neighbor_count: 高頻度とみなす近い n-gram の出現回数の下限。
    """
    bodies = _bodies(pages)
    counts: Counter[str] = Counter()
    for body in bodies.values():
        counts.update(gram for _, gram in _ngrams(body, n))
    # (置換位置, その位置を除いた文字列) → 置換位置に入る文字ごとの出現回数
    buckets: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for gram, count in counts.items():
        for i in range(n):
            buckets[(i, gram[:i] + gram[i + 1 :])][gram[i]] += count

    candidates: list[Candidate] = []
    for page, body in sorted(bodies.items()):
        windows: list[tuple[int, str, str]] = []
        for start, gram in _ngrams(body, n):
            if counts[gram] > max_rare_count:
                continue
            neighbor = _best_hiragana_neighbor(gram, buckets, min_neighbor_count)
            if neighbor is not None:
                windows.append((start, gram, neighbor))
        candidates.extend(_merge_windows(page, body, windows, counts))
    return candidates


def evaluate_page(
    truth_markdown: str, pred_markdown: str, candidates: Iterable[Candidate]
) -> PageScore:
    """1 ページ分の候補を、正解との文字差分で採点する。

    Args:
        truth_markdown: 正解 Markdown。
        pred_markdown: 評価対象の Markdown（検出器に入力したもの）。
        candidates: このページの候補（位置は pred_markdown の body_text 上）。
    """
    spans = _misread_spans(body_text(truth_markdown), body_text(pred_markdown))
    items = list(candidates)
    hits = [c for c in items if any(_overlaps(c.start, c.end, s, e) for s, e in spans)]
    detected = [(s, e) for s, e in spans if any(_overlaps(c.start, c.end, s, e) for c in items)]
    return PageScore(
        flagged=len(items), true_positive=len(hits), misread=len(spans), detected=len(detected)
    )


def _bodies(pages: Mapping[int, str]) -> dict[int, str]:
    return {page: body_text(markdown) for page, markdown in pages.items()}


def _misread_spans(truth: str, pred: str) -> list[tuple[int, int]]:
    """pred の body_text 上で、truth と一致しない区間を返す（空白だけの違いは除く）。

    pred に無い文字（脱落）は、脱落位置の前後 1 文字を含む区間にする。
    """
    spans: list[tuple[int, int]] = []
    matcher = SequenceMatcher(None, pred, truth, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or not (pred[i1:i2] + truth[j1:j2]).strip(_SEPARATOR):
            continue
        if i1 == i2:
            spans.append((max(i1 - 1, 0), min(i1 + 1, len(pred))))
        else:
            spans.append((i1, i2))
    return spans


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _count_words(bodies: Mapping[int, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for body in bodies.values():
        counts.update(m.group(0) for m in _WORD.finditer(body))
    return counts


def _word_candidates(
    bodies: Mapping[int, str],
    counts: Counter[str],
    neighbors: _NeighborIndex,
    kind: CandidateKind,
    max_count: int,
    min_length: int,
    *,
    exclude: Collection[str] = (),
) -> list[Candidate]:
    """出現回数が max_count 以下の語ごとに近い語を探し、出現箇所ごとの候補にする。"""
    excluded = set(exclude)
    suggestions: dict[str, str] = {}
    for word, count in counts.items():
        if count > max_count or len(word) < min_length or word in excluded:
            continue
        near = neighbors.find(word)
        if near:
            # 本文での出現回数が最も多い語を代表にする（同数なら文字列順で決定的にする）
            suggestions[word] = max(sorted(near), key=lambda w: counts[w])
    candidates: list[Candidate] = []
    for page, body in sorted(bodies.items()):
        for match in _WORD.finditer(body):
            word = match.group(0)
            suggestion = suggestions.get(word)
            if suggestion is None:
                continue
            candidates.append(
                Candidate(
                    page=page,
                    kind=kind,
                    surface=word,
                    suggestion=suggestion,
                    count=counts[word],
                    suggestion_count=counts[suggestion],
                    start=match.start(),
                    end=match.end(),
                )
            )
    return candidates


class _NeighborIndex:
    """辞書の語から、編集距離 1（語頭・語末の付け外しを除く）の語を引く索引。

    辞書の語とその 1 文字削除形を鍵にした表を持つ。語 w と辞書の語 d が編集距離 1 なら、
    w 自身か w の 1 文字削除形のどれかが d の鍵に一致するため、候補を絞ってから厳密に判定できる。
    """

    def __init__(self, words: Collection[str]) -> None:
        self._table: dict[str, set[str]] = defaultdict(set)
        for word in words:
            self._table[word].add(word)
            for i in range(len(word)):
                self._table[word[:i] + word[i + 1 :]].add(word)

    def find(self, word: str) -> set[str]:
        """word と編集距離 1 の辞書の語を返す。"""
        keys = {word} | {word[:i] + word[i + 1 :] for i in range(len(word))}
        found: set[str] = set()
        for key in keys:
            found |= self._table.get(key, set())
        return {w for w in found if _is_interior_edit(word, w)}


def _is_interior_edit(a: str, b: str) -> bool:
    """a と b が編集距離 1 で、その編集が語頭・語末への付け外しだけではないかを返す。"""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    if abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    if longer.startswith(shorter) or longer.endswith(shorter):
        return False
    return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))


def _ngrams(body: str, n: int) -> Iterator[tuple[int, str]]:
    """body_text の区切りをまたがない n 文字の窓を、開始位置とともに返す。"""
    for segment in _SEGMENT.finditer(body):
        text = segment.group(0)
        for i in range(len(text) - n + 1):
            yield segment.start() + i, text[i : i + n]


def _best_hiragana_neighbor(
    gram: str, buckets: Mapping[tuple[int, str], Counter[str]], min_neighbor_count: int
) -> str | None:
    """gram とひらがな 1 文字だけ違う高頻度の n-gram のうち、最も多いものを返す。"""
    best: tuple[int, str] | None = None
    for i, char in enumerate(gram):
        if not _HIRAGANA.fullmatch(char):
            continue
        for other, count in buckets[(i, gram[:i] + gram[i + 1 :])].items():
            if other == char or count < min_neighbor_count or not _HIRAGANA.fullmatch(other):
                continue
            if best is None or count > best[0]:
                best = (count, gram[:i] + other + gram[i + 1 :])
    return None if best is None else best[1]


def _merge_windows(
    page: int, body: str, windows: list[tuple[int, str, str]], counts: Counter[str]
) -> list[Candidate]:
    """重なる n-gram の窓を 1 件の候補にまとめる（誤字 1 つが最大 n 個の窓にまたがるため）。

    windows は (開始位置, n-gram, 近い高頻度 n-gram) を開始位置の昇順に並べたもの。
    まとめた候補の suggestion と count は、先頭の窓のものを使う。
    """
    groups: list[list[tuple[int, str, str]]] = []
    for window in windows:
        if groups and window[0] < groups[-1][-1][0] + len(groups[-1][-1][1]):
            groups[-1].append(window)
        else:
            groups.append([window])
    candidates: list[Candidate] = []
    for group in groups:
        first_start, first_gram, first_neighbor = group[0]
        last_start, last_gram, _ = group[-1]
        end = last_start + len(last_gram)
        candidates.append(
            Candidate(
                page=page,
                kind="rare_ngram",
                surface=body[first_start:end],
                suggestion=first_neighbor,
                count=counts[first_gram],
                suggestion_count=counts[first_neighbor],
                start=first_start,
                end=end,
            )
        )
    return candidates


app = typer.Typer(add_completion=False)

_KIND_LABELS: dict[str, str] = {
    "rare_word": "低頻度語",
    "index_near": "索引語に近い語",
    "rare_ngram": "かな n-gram",
}


@app.command()
def consistency(
    pred: Annotated[
        Path, typer.Option("--pred", help="評価対象（--no-combine 出力 または ページキャッシュ）")
    ],
    truth: Annotated[
        Path | None,
        typer.Option("--truth", help="評価セット（指定すると適合率・再現率を計算する）"),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="検出結果を書き出す JSON ファイル")
    ] = None,
    skip_index: Annotated[
        bool, typer.Option("--skip-index", help="索引ページが無い本で索引語の検出を省く")
    ] = False,
) -> None:
    """書籍内の表記の一貫性から、地の文の誤字候補を検出して表示する。"""
    try:
        predictions = {page: read_prediction(pred, page) for page in list_prediction_pages(pred)}
        index_pages = detect_index_pages(predictions)
        if not index_pages and not skip_index:
            raise ValueError(
                "索引ページを検出できませんでした（索引の無い本は --skip-index を指定してください）"
            )
        body_pages = {p: md for p, md in predictions.items() if p not in index_pages}
        index_terms: set[str] = set()
        for page in index_pages:
            index_terms |= parse_index_terms(predictions[page])
        candidates = [
            *find_rare_word_candidates(body_pages),
            *find_index_near_candidates(body_pages, index_terms),
            *find_rare_ngram_candidates(body_pages),
        ]
        scores = _score(truth, pred, predictions, candidates) if truth is not None else None
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    console = Console()
    console.print(
        f"ページ数 {len(predictions)}（索引ページ {len(index_pages)}・索引語 {len(index_terms)}）"
    )
    console.print(_candidate_table(candidates))
    if scores is not None:
        console.print(_score_table(scores))
    if json_path is not None:
        result: dict[str, Any] = {
            "page_count": len(predictions),
            "index_pages": sorted(index_pages),
            "index_term_count": len(index_terms),
            "candidates": [c.to_dict() for c in candidates],
        }
        if scores is not None:
            result["scores"] = scores
        try:
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            typer.echo(f"エラー: {e}", err=True)
            raise typer.Exit(code=1) from e
        console.print(f"検出結果を書き出しました: {json_path}")


def _score(
    truth: Path, pred: Path, predictions: Mapping[int, str], candidates: list[Candidate]
) -> dict[str, dict[str, int | float]]:
    """評価セットのページについて、検出器の種類ごと（と全体）の適合率・再現率を計算する。"""
    entries = load_manifest(truth)
    for entry in entries:
        if entry.page not in predictions:
            raise FileNotFoundError(
                f"評価セットのページ {entry.page} が評価対象にありません: {pred}"
            )
    groups: dict[str, list[Candidate]] = {kind: [] for kind in _KIND_LABELS}
    for candidate in candidates:
        groups[candidate.kind].append(candidate)
    groups["all"] = candidates
    result: dict[str, dict[str, int | float]] = {}
    for name, group in groups.items():
        page_scores = [
            evaluate_page(
                read_truth(truth, entry.page),
                predictions[entry.page],
                [c for c in group if c.page == entry.page],
            )
            for entry in entries
        ]
        result[name] = sum_page_scores(page_scores).to_dict()
    return result


def _candidate_table(candidates: list[Candidate]) -> Table:
    counts: Counter[str] = Counter(c.kind for c in candidates)
    summary = "・".join(f"{label} {counts[kind]}" for kind, label in _KIND_LABELS.items())
    table = Table(title=f"誤字候補（{len(candidates)} 件: {summary}）")
    for column in ("ページ", "種類", "表記", "近い表記", "表記の回数", "近い表記の回数"):
        table.add_column(column)
    for c in candidates:
        table.add_row(
            f"p.{c.page}",
            _KIND_LABELS[c.kind],
            c.surface,
            c.suggestion,
            str(c.count),
            str(c.suggestion_count),
        )
    return table


def _score_table(scores: Mapping[str, Mapping[str, int | float]]) -> Table:
    table = Table(title="評価セットでの適合率・再現率（正解との文字差分を誤読とみなす）")
    for column in ("検出器", "候補", "うち的中", "誤読箇所", "検出された誤読", "適合率", "再現率"):
        table.add_column(column)
    for name, s in scores.items():
        table.add_row(
            _KIND_LABELS.get(name, "全体"),
            str(s["flagged"]),
            str(s["true_positive"]),
            str(s["misread"]),
            str(s["detected"]),
            f"{s['precision']:.3f}",
            f"{s['recall']:.3f}",
        )
    return table


if __name__ == "__main__":
    app()
