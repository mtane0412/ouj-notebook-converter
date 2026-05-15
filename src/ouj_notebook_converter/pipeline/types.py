"""仕様: パイプライン各段の入出力データ型を定義する。

- 不変オブジェクト中心（frozen=True の dataclass）。
- I/O は持たない（pure data）。
- ページ番号は 0-origin（PDF の内部インデックスに合わせる）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PageJob:
    """1 ページ分の処理単位。パイプラインへの入力情報を保持する。"""

    pdf_path: Path
    page_index: int       # 0-origin
    dpi: int
    cache_key: str        # SHA-256 ハッシュ（cache/key.py が算出）


@dataclass(frozen=True)
class PageAnalysis:
    """analyze ステージの出力（ディスクキャッシュの対象）。

    yomitoku_json_path と markdown_raw_path は実際にファイルが存在することを
    保証しない。存在確認はキャッシュストアが行う。
    """

    page_index: int
    yomitoku_json_path: Path           # results.to_json() の書き込み先
    figure_paths: list[Path] = field(default_factory=list)  # 切り出し figure 画像
    markdown_raw_path: Path = Path()   # results.to_markdown() のそのままの出力


@dataclass(frozen=True)
class MathOverlay:
    """figure 画像 → LaTeX 候補の対応表。math_extract ステージの出力。

    items の値が空文字列の場合は「変換しなかった（数式ではない）」を意味する。
    """

    items: dict[Path, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PageMarkdown:
    """assemble ステージの出力。1 ページ分の最終的な Markdown 文字列と参照アセット。"""

    page_index: int
    markdown: str
    referenced_assets: list[Path] = field(default_factory=list)
