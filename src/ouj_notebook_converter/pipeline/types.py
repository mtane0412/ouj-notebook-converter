"""仕様: パイプライン各段の入出力データ型を定義する。

- 不変オブジェクト中心（frozen=True の dataclass）。
- I/O は持たない（pure data）。
- ページ番号は 0-origin（PDF の内部インデックスに合わせる）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class PageJob:
    """1 ページ分の処理単位。パイプラインへの入力情報を保持する。"""

    pdf_path: Path
    page_index: int  # 0-origin
    dpi: int
    cache_key: str  # SHA-256 ハッシュ（cache/key.py が算出）


@dataclass(frozen=True)
class PageAnalysis:
    """analyze ステージの出力（ディスクキャッシュの対象）。

    yomitoku_json_path と markdown_raw_path は実際にファイルが存在することを
    保証しない。存在確認はキャッシュストアが行う。
    """

    page_index: int
    yomitoku_json_path: Path  # results.to_json() の書き込み先
    figure_paths: list[Path] = field(default_factory=list)  # 切り出し figure 画像
    markdown_raw_path: Path = Path()  # results.to_markdown() のそのままの出力


@dataclass(frozen=True)
class MathOverlay:
    """math_extract ステージの出力。クロップ画像 → LaTeX の対応を保持する。

    items[crop_path]    : LaTeX 文字列。空文字は「変換しなかった」を意味する
    roles[crop_path]    : "inline_formula" / "display_formula"
    originals[crop_path]: 元 paragraph.contents（raw.md 上の置換対象テキスト）
    """

    items: dict[Path, str] = field(default_factory=dict)
    roles: dict[Path, str] = field(default_factory=dict)
    originals: dict[Path, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PageMarkdown:
    """assemble ステージの出力。1 ページ分の最終的な Markdown 文字列と参照アセット。"""

    page_index: int
    markdown: str
    referenced_assets: list[Path] = field(default_factory=list)


class ChapterKind(str, Enum):
    """章の種別。"""

    PREFACE = "preface"  # 前書き / まえがき / はじめに
    CHAPTER = "chapter"  # 第N章
    AFTERWORD = "afterword"  # 後書き / あとがき / おわりに
    INDEX = "index"  # 索引


@dataclass(frozen=True)
class ChapterSpec:
    """章検出ステージの出力。1 章分のメタ情報とページ範囲を保持する。

    - order: 出力ファイル名のゼロパディング番号（00=前書き, 01〜N=本章, N+1=後書き, N+2=索引）
    - kind: 章の種別
    - chapter_number: 第N章のN（kind=CHAPTER のときのみ非 None）
    - title: 章見出し本文（例: 「データとは何か」）
    - start_page_index: 0-origin で章が始まるページ（inclusive）
    - end_page_index: 0-origin で章が終わるページ（inclusive）
    - source: 検出ソース識別子 ("syllabus" / "pdf_toc" / "ocr_toc" / "body_headings")
    """

    order: int
    kind: ChapterKind
    chapter_number: int | None
    title: str
    start_page_index: int
    end_page_index: int
    source: str
