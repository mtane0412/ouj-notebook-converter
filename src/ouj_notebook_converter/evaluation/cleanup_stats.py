"""仕様: OCR 出力に対して後処理（normalize_ocr_markdown）の各ルールが何回発火するかを数える。

プロンプト改善（#26）の評価用。プロンプトで出力形式を指示した結果、後処理が必要だった
破綻がどれだけ減ったか（発生源で抑えられたか）を、ページ Markdown の生の出力から数える。

使い方:
  uv run python -m ouj_notebook_converter.evaluation.cleanup_stats --pred <出力ディレクトリ>

出力: ルール名ごとの件数と、後処理で直せたものの合計 `total` を JSON で標準出力に書く。

注意事項:
  - 件数は markdown_cleanup のルール（正規表現）と同じ判定で数える。ルールを追加・変更した
    ときは本モジュールも合わせること
  - コードフェンス内は後処理の対象外なので数えない
  - `unfixable_*` と `empty_math` は後処理で直せない（または直さない）破綻。`total` には含めない
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated

import typer

from ouj_notebook_converter.pipeline.stages import markdown_cleanup as mc

app = typer.Typer(add_completion=False)

_UNFIXABLE_CLINE = re.compile(r"\\cline\b")
_UNFIXABLE_MULTICOLUMN = re.compile(r"\\multicolumn\b")
_UNFIXABLE_ENCLOSE = re.compile(r"\\enclose\b")
_EQNARRAY_STAR_BEGIN = re.compile(r"\\begin\{eqnarray\*\}")
_EQNARRAY_BEGIN = re.compile(r"\\begin\{eqnarray\}")
_EMPTY_MATH_BODY = re.compile(r"^(\$\$|\$)\s*(\$\$|\$)$")

# 後処理が直せた破綻（total に含める）
_FIXED_RULES = (
    "hfill",
    "tag",
    "tag_only_math",
    "quad",
    "eqnarray_star",
    "array_column_spacing",
    "section_heading_level",
    "label_heading",
)


def _count_chunk(text: str, counts: Counter[str]) -> None:
    """コードフェンスを含まない断片の発火件数を counts に加える。"""
    counts["label_heading"] += len(mc._LABEL_HEADING.findall(text))
    # ラベル見出しは先に太字へ戻るため、節見出しの判定は変換後の文字列に対して行う
    after_label = mc._LABEL_HEADING.sub(mc._label_heading_to_bold, text)
    counts["section_heading_level"] += len(mc._SECTION_HEADING.findall(after_label))

    last_end = 0
    for math in mc.MATH_SPAN.finditer(after_label):
        _count_prose(after_label[last_end : math.start()], counts)
        _count_math(math.group(0), counts)
        last_end = math.end()
    _count_prose(after_label[last_end:], counts)


def _count_prose(text: str, counts: Counter[str]) -> None:
    counts["hfill"] += len(mc._HFILL.findall(text))
    counts["tag"] += len(mc._TAG.findall(text))
    counts["quad"] += len(mc._QQUAD.findall(text)) + len(mc._QUAD.findall(text))


def _count_math(span: str, counts: Counter[str]) -> None:
    counts["eqnarray_star"] += len(_EQNARRAY_STAR_BEGIN.findall(span))
    for column_spec in mc._ARRAY_COLUMN_SPEC.finditer(span):
        if mc._COLUMN_SPACING.search(column_spec.group(2)):
            counts["array_column_spacing"] += 1
    delimited = mc._MATH_DELIMITERS.match(span)
    if delimited is not None and mc._TAG_ONLY_BODY.match(delimited.group(2)):
        counts["tag_only_math"] += 1
    if _EMPTY_MATH_BODY.match(span):
        counts["empty_math"] += 1
    counts["unfixable_cline"] += len(_UNFIXABLE_CLINE.findall(span))
    counts["unfixable_multicolumn"] += len(_UNFIXABLE_MULTICOLUMN.findall(span))
    counts["unfixable_enclose"] += len(_UNFIXABLE_ENCLOSE.findall(span))
    counts["unfixable_eqnarray_numbered"] += len(_EQNARRAY_BEGIN.findall(span))


def count_cleanup_firings(markdown: str) -> dict[str, int]:
    """1 ページ分の Markdown に対する後処理ルールごとの発火件数を数える。

    Args:
        markdown: 後処理前（生）のページ Markdown。

    Returns:
        ルール名から件数への辞書。全ルールのキーを含み、発火が無ければ 0。
    """
    counts: Counter[str] = Counter()
    for segment, in_fence in mc.split_code_fences(markdown):
        if not in_fence:
            _count_chunk(segment, counts)
    keys = (
        *_FIXED_RULES,
        "empty_math",
        "unfixable_cline",
        "unfixable_multicolumn",
        "unfixable_enclose",
        "unfixable_eqnarray_numbered",
    )
    return {key: counts[key] for key in keys}


def sum_firings(per_page: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """ページごとの件数をルール名ごとに合算する。"""
    total: Counter[str] = Counter()
    for counts in per_page:
        total.update(counts)
    return dict(total)


@app.command()
def main(
    pred: Annotated[Path, typer.Option(help="page_NNNN/raw.md を含む出力ディレクトリ")],
) -> None:
    """出力ディレクトリ全体の後処理発火件数を JSON で標準出力に書く。"""
    per_page = [
        count_cleanup_firings(raw.read_text(encoding="utf-8"))
        for raw in sorted(pred.glob("page_*/raw.md"))
    ]
    if not per_page:
        raise typer.BadParameter(f"page_NNNN/raw.md が見つかりません: {pred}")
    total = sum_firings(per_page)
    total["total"] = sum(total[rule] for rule in _FIXED_RULES)
    typer.echo(json.dumps(total, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
