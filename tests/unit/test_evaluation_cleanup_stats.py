"""仕様: evaluation.cleanup_stats（後処理の発火件数の計測）のテスト。

normalize_ocr_markdown の各ルールが 1 ページの Markdown に対して何回発火するかを数える。
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ouj_notebook_converter.evaluation.cleanup_stats import (
    app,
    count_cleanup_firings,
    sum_firings,
)


def test_破綻の無いMarkdownは全ルールが0件になる() -> None:
    counts = count_cleanup_firings("# 1　数の概念\n\n## 1.1　自然数 (A)\n\n本文です。$x+1$\n")
    assert sum(counts.values()) == 0


def test_hfillとtagとquadを地の文でそれぞれ数える() -> None:
    markdown = "式を示す。 \\hfill (1.18)\n\n結果 \\tag{2.3} です。\n\nＡ\\quad Ｂ\\qquad Ｃ\n"
    counts = count_cleanup_firings(markdown)
    assert counts["hfill"] == 1
    assert counts["tag"] == 1
    assert counts["quad"] == 2


def test_本体がtagだけの数式を数える() -> None:
    counts = count_cleanup_firings("$$\\tag{3.1}$$\n")
    assert counts["tag_only_math"] == 1


def test_数式内のeqnarrayとarray列指定を数える() -> None:
    markdown = (
        "$$\\begin{eqnarray*} a &=& b \\end{eqnarray*}$$\n\n"
        "$$\\begin{array}{r@{\\,}l} a & b \\end{array}$$\n"
    )
    counts = count_cleanup_firings(markdown)
    assert counts["eqnarray_star"] == 1
    assert counts["array_column_spacing"] == 1


def test_節見出しのレベル揺れと例ラベルの見出し化を数える() -> None:
    markdown = "### 4.5 $n$ 乗根の大小 (C)\n\n### 例 4.4\n\n## 4.6 指数法則 (A)\n"
    counts = count_cleanup_firings(markdown)
    assert counts["section_heading_level"] == 1
    assert counts["label_heading"] == 1


def test_コードフェンス内は数えない() -> None:
    counts = count_cleanup_firings("```\n\\hfill (1.1)\n### 例 1.1\n```\n")
    assert sum(counts.values()) == 0


def test_変換できない残骸は別枠で数える() -> None:
    markdown = "$$\\begin{array}{c|c} \\cline{2-3} \\multicolumn{2}{c}{a} \\end{array}$$\n\n$$ $$\n"
    counts = count_cleanup_firings(markdown)
    assert counts["unfixable_cline"] == 1
    assert counts["unfixable_multicolumn"] == 1
    assert counts["empty_math"] == 1


def test_sum_firingsはページごとの件数を合算する() -> None:
    total = sum_firings([{"hfill": 1, "tag": 0}, {"hfill": 2, "tag": 3}])
    assert total == {"hfill": 3, "tag": 3}


def test_CLIはpredディレクトリ全体の件数をJSONで出す(tmp_path: Path) -> None:
    for page, body in ((3, "a \\hfill (1.1)\n"), (4, "### 例 2.1\n")):
        page_dir = tmp_path / f"page_{page:04d}"
        page_dir.mkdir()
        (page_dir / "raw.md").write_text(body, encoding="utf-8")
    result = CliRunner().invoke(app, ["--pred", str(tmp_path)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["hfill"] == 1
    assert data["label_heading"] == 1
    assert data["total"] == 2
