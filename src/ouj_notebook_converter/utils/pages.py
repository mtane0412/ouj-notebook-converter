"""仕様: ページ範囲文字列（"1,3-5,10" 形式）を整数リストにパースするユーティリティ。"""

from __future__ import annotations


def parse_page_range(spec: str | None, *, total: int) -> list[int]:
    """ページ範囲指定文字列を 1-origin のページ番号リストに変換する。

    Args:
        spec: "1,3-5,10" 形式のページ範囲。None の場合は全ページ [1..total] を返す。
        total: PDF の総ページ数。

    Returns:
        重複なし・昇順のページ番号リスト（1-origin）。

    Raises:
        ValueError: 構文エラー、範囲外ページ、逆順範囲の場合。
    """
    if spec is None:
        return list(range(1, total + 1))

    if not spec.strip():
        raise ValueError("ページ範囲の指定が空です。例: '1,3-5,10'")

    pages: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"ページ範囲の書式が不正です: '{token}'") from exc
            if start > end:
                raise ValueError(f"範囲の開始が終了より大きいです: '{token}'")
            if start < 1:
                raise ValueError(f"ページ番号は1以上でなければなりません: '{token}'")
            if end > total:
                raise ValueError(f"ページ番号 {end} が total={total} を超えています: '{token}'")
            pages.update(range(start, end + 1))
        else:
            try:
                page = int(token)
            except ValueError as exc:
                raise ValueError(f"ページ範囲の書式が不正です: '{token}'") from exc
            if page < 1:
                raise ValueError(f"ページ番号は1以上でなければなりません: '{token}'")
            if page > total:
                raise ValueError(f"ページ番号 {page} が total={total} を超えています: '{token}'")
            pages.add(page)

    return sorted(pages)
