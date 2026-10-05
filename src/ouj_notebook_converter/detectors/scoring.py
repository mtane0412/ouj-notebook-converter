"""仕様: 検出器が誤読候補にした数式を評価セットの正解と照らし合わせ、適合率・再現率を数える。

定義（数式 1 つを単位とする）:
  - 誤読: 評価対象の数式のうち、正解の数式と多重集合として一致しないもの
    （比較は evaluation の数式F1 と同じく normalize_latex で表記の違いを除いて行う）
  - 適合率 = 誤読候補のうち誤読だった件数 / 誤読候補の件数
  - 再現率 = 誤読候補のうち誤読だった件数 / 誤読の件数

注意事項:
  - 誤読には検算で検出しようのないもの（括弧の過不足など）も含まれるため、
    再現率は「数式の誤読全体のうち、この検出器で見つかる割合」を表す
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Set
from dataclasses import dataclass

from ouj_notebook_converter.evaluation.markdown_parts import normalize_latex, split_markdown


@dataclass(frozen=True)
class DetectionScore:
    """誤読候補と誤読の件数。

    Attributes:
        flagged: 誤読候補にした数式の件数。
        true_positive: 誤読候補のうち、誤読だった数式の件数。
        misread: 誤読の件数（正解と一致しない評価対象の数式の件数）。
    """

    flagged: int
    true_positive: int
    misread: int

    @property
    def precision(self) -> float:
        """適合率。誤読候補が 0 件なら誤検出も無いため 1 とする。"""
        return self.true_positive / self.flagged if self.flagged else 1.0

    @property
    def recall(self) -> float:
        """再現率。誤読が 0 件なら見落としも無いため 1 とする。"""
        return self.true_positive / self.misread if self.misread else 1.0

    def to_dict(self) -> dict[str, int | float]:
        """JSON に書き出すための辞書を返す。"""
        return {
            "flagged": self.flagged,
            "true_positive": self.true_positive,
            "misread": self.misread,
            "precision": self.precision,
            "recall": self.recall,
        }


def score_page(
    truth_markdown: str, pred_markdown: str, flagged_formula_indices: Set[int]
) -> DetectionScore:
    """1 ページ分の誤読候補を正解と照らし合わせる。

    Args:
        truth_markdown: 正解 Markdown。
        pred_markdown: 評価対象の Markdown（検出器に入力したもの）。
        flagged_formula_indices: 誤読候補にした数式の、評価対象での出現順の番号（0 始まり）。

    Returns:
        件数。
    """
    truth = Counter(_normalized(f.tex for f in split_markdown(truth_markdown).formulas))
    pred_formulas = split_markdown(pred_markdown).formulas
    misread = Counter(_normalized(f.tex for f in pred_formulas)) - truth
    flagged = Counter(_normalized(pred_formulas[i].tex for i in sorted(flagged_formula_indices)))
    return DetectionScore(
        flagged=sum(flagged.values()),
        true_positive=sum((flagged & misread).values()),
        misread=sum(misread.values()),
    )


def sum_scores(scores: Iterable[DetectionScore]) -> DetectionScore:
    """複数ページの件数を合計する。"""
    items = list(scores)
    return DetectionScore(
        flagged=sum(s.flagged for s in items),
        true_positive=sum(s.true_positive for s in items),
        misread=sum(s.misread for s in items),
    )


def _normalized(texes: Iterable[str]) -> list[str]:
    """数式を正規化する。日本語だけの数式は正規化後に空になるため除く（数式F1 と同じ扱い）。"""
    return [tex for tex in (normalize_latex(t) for t in texes) if tex]
