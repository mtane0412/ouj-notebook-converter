"""仕様: 人手で確認した OCR 修正を、キャッシュ外の修正ファイルとして管理・適用する。

修正ファイル（JSON）の形式:

    {
      "version": 1,
      "corrections": [
        {"page": 70, "before": "置換前", "after": "置換後",
         "reason": "任意", "reviewer": "任意", "date": "任意"}
      ]
    }

- page は 1 始まり（キャッシュの page_NNNN と同じ番号）。
- before は当該ページの OCR Markdown（raw.md）中にちょうど 1 か所だけ現れる文字列。
  見つからない場合・複数見つかる場合は CorrectionError で停止する（Fail-Fast）。
- 同一ページの修正は記載順に適用する。
- キャッシュ（raw.md）は書き換えない。変換時にメモリ上のテキストへ適用する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_VERSION = 1
_REQUIRED_KEYS = {"page", "before", "after"}
_OPTIONAL_KEYS = {"reason", "reviewer", "date"}


class CorrectionError(ValueError):
    """修正ファイルの形式不正、または修正の適用失敗を表す例外。"""


@dataclass(frozen=True)
class Correction:
    """1 件の人手修正。

    page は 1 始まりのページ番号。reason / reviewer / date は記録用で適用結果に影響しない。
    """

    page: int
    before: str
    after: str
    reason: str = ""
    reviewer: str = ""
    date: str = ""


def _parse_entry(index: int, entry: Any) -> Correction:
    """修正ファイルの 1 エントリを検証して Correction に変換する。"""
    where = f"corrections[{index}]"
    if not isinstance(entry, dict):
        raise CorrectionError(f"{where}: オブジェクトである必要があります")
    unknown = set(entry) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise CorrectionError(f"{where}: 未知のキーがあります: {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(entry)
    if missing:
        raise CorrectionError(f"{where}: 必須キーがありません: {sorted(missing)}")

    page = entry["page"]
    # bool は int のサブクラスなので明示的に除外する
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise CorrectionError(f"{where}: page は 1 以上の整数で指定してください: {page!r}")
    for key in sorted(_REQUIRED_KEYS - {"page"} | _OPTIONAL_KEYS):
        if key in entry and not isinstance(entry[key], str):
            raise CorrectionError(f"{where}: {key} は文字列で指定してください")
    if entry["before"] == "":
        raise CorrectionError(f"{where}: before は空文字にできません")
    if entry["before"] == entry["after"]:
        raise CorrectionError(f"{where}: before と after が同じです（修正になっていません）")

    return Correction(
        page=page,
        before=entry["before"],
        after=entry["after"],
        reason=entry.get("reason", ""),
        reviewer=entry.get("reviewer", ""),
        date=entry.get("date", ""),
    )


def load_corrections(path: Path) -> dict[int, tuple[Correction, ...]]:
    """修正ファイルを読み込み、ページ番号（1 始まり）ごとの修正タプルを返す。

    Raises:
        CorrectionError: ファイルが無い、JSON が不正、形式が不正な場合。
    """
    if not path.exists():
        raise CorrectionError(f"修正ファイルが見つかりません: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorrectionError(f"修正ファイルを JSON として読み込めません: {path}: {e}") from e

    if not isinstance(data, dict) or data.get("version") != SUPPORTED_VERSION:
        raise CorrectionError(
            f"修正ファイルの version は {SUPPORTED_VERSION} である必要があります: {path}"
        )
    entries = data.get("corrections")
    if not isinstance(entries, list):
        raise CorrectionError(f"修正ファイルの corrections は配列である必要があります: {path}")

    by_page: dict[int, list[Correction]] = {}
    for i, entry in enumerate(entries):
        correction = _parse_entry(i, entry)
        by_page.setdefault(correction.page, []).append(correction)
    return {page: tuple(items) for page, items in by_page.items()}


def apply_corrections(text: str, corrections: tuple[Correction, ...]) -> str:
    """1 ページ分のテキストに修正を記載順に適用する。

    Raises:
        CorrectionError: before が見つからない、または複数一致する場合。
    """
    for c in corrections:
        count = text.count(c.before)
        if count == 0:
            raise CorrectionError(
                f"{c.page} ページの修正で置換前の文字列が見つかりません: {c.before!r}"
            )
        if count > 1:
            raise CorrectionError(
                f"{c.page} ページの修正で置換前の文字列が {count} か所に一致しました"
                f"（一意になる長さに延ばしてください）: {c.before!r}"
            )
        text = text.replace(c.before, c.after, 1)
    return text
