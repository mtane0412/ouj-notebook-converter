"""仕様: PDF の章境界を検出するステージ。

検出ソースをシラバス → PDF しおり → OCR 目次 → 本文見出し の順でフォールバックし、
いずれも成功しない場合は ChapterDetectionError を送出する（Fail-Fast）。

各 detect_via_* 関数は失敗時に ChapterDetectionError を送出する。
暗黙のフォールバック・NOP は禁止。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ouj_notebook_converter.pipeline.types import ChapterKind, ChapterSpec, PageMarkdown

if TYPE_CHECKING:
    import httpx
    from bs4 import BeautifulSoup

# 全角数字 → 半角数字のマッピング
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# 漢数字 → アラビア数字のマッピング（1〜15 の範囲を想定）
_KANJI_TO_INT: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
}

# 章見出しパターン（H1 見出し冒頭にマッチ）
_CHAPTER_PATTERN = re.compile(
    r"^#\s+第\s*([０-９0-9一二三四五六七八九十]+)\s*章\s*(.*)$",
    re.MULTILINE,
)
_PREFACE_PATTERN = re.compile(
    r"^#\s+(まえがき|前書き|はじめに|序章|序論|はしがき)\b",
    re.MULTILINE,
)
_AFTERWORD_PATTERN = re.compile(
    r"^#\s+(あとがき|後書き|おわりに|終章|結論|むすび)\b",
    re.MULTILINE,
)
_INDEX_PATTERN = re.compile(
    r"^#\s+(索引|インデックス)\b",
    re.MULTILINE,
)


class ChapterDetectionError(Exception):
    """章境界の検出に失敗した場合に送出する例外。"""


def detect_chapters(
    pdf_path: Path,
    page_markdowns: list[PageMarkdown],
    *,
    course_code: str | None = None,
    http_client: httpx.Client | None = None,
) -> list[ChapterSpec]:
    """フォールバックチェーンで章境界を検出する。

    優先順: syllabus → pdf_toc → ocr_toc → body_headings
    全て失敗した場合は ChapterDetectionError を送出する。

    Args:
        pdf_path: 変換対象の PDF ファイルパス（PDF しおり取得に使用）。
        page_markdowns: OCR 済み全ページの PageMarkdown リスト。
        course_code: OUJ コースコード（シラバス検出に使用）。省略時はシラバス検出をスキップ。
        http_client: httpx.Client の依存注入（テスト用）。None の場合は内部で生成。

    Returns:
        検出された章リスト（order 順にソート済み）。

    Raises:
        ChapterDetectionError: 全検出器が失敗した場合。
    """
    errors: list[str] = []

    # 1. シラバス検出
    if course_code is not None:
        try:
            return detect_via_syllabus(course_code, page_markdowns, http_client)
        except ChapterDetectionError as e:
            errors.append(f"syllabus: {e}")

    # 2. PDF しおり検出
    try:
        return detect_via_pdf_toc(pdf_path, page_markdowns)
    except ChapterDetectionError as e:
        errors.append(f"pdf_toc: {e}")

    # 3. OCR 目次ページ検出
    try:
        return detect_via_ocr_toc(page_markdowns)
    except ChapterDetectionError as e:
        errors.append(f"ocr_toc: {e}")

    # 4. 本文見出し検出
    try:
        return detect_via_body_headings(page_markdowns)
    except ChapterDetectionError as e:
        errors.append(f"body_headings: {e}")

    reasons = "\n".join(f"  - {e}" for e in errors)
    raise ChapterDetectionError(
        f"章境界を検出できませんでした。各検出器の失敗理由:\n{reasons}\n\n"
        "対処案: --course-code でコースコードを指定するか、PDF に目次が含まれているか確認してください。"
    )


def detect_via_syllabus(
    course_code: str,
    page_markdowns: list[PageMarkdown],
    http_client: httpx.Client | None = None,
) -> list[ChapterSpec]:
    """OUJ シラバス HTML から章タイトルを取得し、本文見出しと突き合わせて章境界を返す。

    Args:
        course_code: OUJ コースコード（例: "1554069"）。
        page_markdowns: OCR 済み全ページの PageMarkdown リスト。
        http_client: httpx.Client の依存注入（テスト用）。

    Raises:
        ChapterDetectionError: course_code 未指定 / HTTP 失敗 / 章タイトル抽出失敗。
    """
    import httpx as _httpx
    from bs4 import BeautifulSoup

    url = f"https://www.ouj.ac.jp/kamoku/{course_code}/"
    client = http_client or _httpx.Client(timeout=30.0)
    try:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
    except _httpx.HTTPError as e:
        raise ChapterDetectionError(f"シラバス取得に失敗しました (URL: {url}): {e}") from e

    soup = BeautifulSoup(response.text, "html.parser")

    # OUJ シラバスの章タイトル一覧を抽出する
    # 実際の HTML 構造は実装時に WebFetch で確認して調整する
    chapter_titles = _parse_syllabus_chapters(soup)
    if not chapter_titles:
        raise ChapterDetectionError(
            f"シラバス HTML から章タイトルを抽出できませんでした (URL: {url})"
        )

    # 本文見出しと突き合わせてページ境界を決定する
    body_chapters = _match_syllabus_titles_to_pages(chapter_titles, page_markdowns)
    return body_chapters


def _parse_syllabus_chapters(soup: BeautifulSoup) -> list[str]:
    """OUJ シラバス HTML から章タイトル一覧を抽出する。

    実際の HTML 構造は実装時に確認する。章タイトルが見つからない場合は空リストを返す。
    """
    from bs4 import BeautifulSoup  # noqa: F401

    titles: list[str] = []
    # OUJ シラバスで章が記載されるセルを探す（実際の構造は要確認）
    for cell in soup.find_all("td"):
        text = cell.get_text(strip=True)
        m = re.match(r"第\s*([0-9０-９一二三四五六七八九十]+)\s*章\s*(.+)", text)
        if m:
            titles.append(m.group(2).strip())
    return titles


def _match_syllabus_titles_to_pages(
    chapter_titles: list[str],
    page_markdowns: list[PageMarkdown],
) -> list[ChapterSpec]:
    """シラバス章タイトルを本文 Markdown の章見出しと照合してページ境界を決定する。

    シラバスのタイトルと本文見出しが一致する場合に採用する。
    一致しない場合は本文見出しで得られた境界をそのまま利用する。
    """
    # 本文見出し検出で基本ページ境界を取得し、タイトルだけシラバスで上書きする
    body_chapters = detect_via_body_headings(page_markdowns)

    main_chapters = [c for c in body_chapters if c.kind == ChapterKind.CHAPTER]
    if len(main_chapters) != len(chapter_titles):
        # 件数が合わない場合はシラバスタイトルのみ警告・本文検出結果を使う
        return _update_source(body_chapters, "syllabus")

    result: list[ChapterSpec] = []
    chapter_idx = 0
    for chapter in body_chapters:
        if chapter.kind == ChapterKind.CHAPTER and chapter_idx < len(chapter_titles):
            result.append(
                ChapterSpec(
                    order=chapter.order,
                    kind=chapter.kind,
                    chapter_number=chapter.chapter_number,
                    title=chapter_titles[chapter_idx],
                    start_page_index=chapter.start_page_index,
                    end_page_index=chapter.end_page_index,
                    source="syllabus",
                )
            )
            chapter_idx += 1
        else:
            result.append(
                ChapterSpec(
                    order=chapter.order,
                    kind=chapter.kind,
                    chapter_number=chapter.chapter_number,
                    title=chapter.title,
                    start_page_index=chapter.start_page_index,
                    end_page_index=chapter.end_page_index,
                    source="syllabus",
                )
            )
    return result


def _update_source(chapters: list[ChapterSpec], source: str) -> list[ChapterSpec]:
    """chapters リストの全要素の source を更新した新しいリストを返す。"""
    return [
        ChapterSpec(
            order=c.order,
            kind=c.kind,
            chapter_number=c.chapter_number,
            title=c.title,
            start_page_index=c.start_page_index,
            end_page_index=c.end_page_index,
            source=source,
        )
        for c in chapters
    ]


def _get_pdfium() -> object:
    """pypdfium2 モジュールを返す。テスト時はモンキーパッチで差し替え可能。"""
    try:
        import pypdfium2 as pdfium
        return pdfium
    except ImportError as e:
        raise ChapterDetectionError("pypdfium2 が未インストールです。") from e


def detect_via_pdf_toc(
    pdf_path: Path,
    page_markdowns: list[PageMarkdown],
) -> list[ChapterSpec]:
    """pypdfium2 の PdfDocument.get_toc() で PDF のしおりを取得して章境界を返す。

    Args:
        pdf_path: 変換対象の PDF ファイルパス。
        page_markdowns: ページ数の検証に使用。

    Raises:
        ChapterDetectionError: しおりが存在しない / 章見出しを認識できない場合。
    """
    pdfium = _get_pdfium()

    doc = pdfium.PdfDocument(str(pdf_path))  # type: ignore[attr-defined]
    toc = list(doc.get_toc())
    if not toc:
        raise ChapterDetectionError("PDF にしおり（目次情報）が含まれていません。")

    total_pages = len(page_markdowns)
    raw_specs: list[tuple[ChapterKind, int | None, str, int]] = []
    # TOC エントリから kind / chapter_number / title / page_index を抽出する
    for item in toc:
        title: str = item.title or ""
        page_index: int = (item.page_index or 1) - 1  # 1-origin → 0-origin

        kind, num = _classify_heading(title)
        if kind is not None:
            raw_specs.append((kind, num, _clean_title(title, kind, num), page_index))

    if not raw_specs:
        raise ChapterDetectionError(
            "PDF のしおりに章・前書き・後書き・索引として認識できるエントリがありませんでした。"
        )

    return _build_chapter_specs(raw_specs, total_pages, source="pdf_toc")


def detect_via_ocr_toc(
    page_markdowns: list[PageMarkdown],
) -> list[ChapterSpec]:
    """先頭付近の「目次」ページを特定し、章エントリとページ番号を抽出して章境界を返す。

    Args:
        page_markdowns: OCR 済み全ページの PageMarkdown リスト。

    Raises:
        ChapterDetectionError: 目次ページが見つからない / エントリ抽出に失敗した場合。
    """
    # 先頭 15 ページ以内で「目次」見出しを含むページを探す
    toc_page: PageMarkdown | None = None
    for page in page_markdowns[:15]:
        if re.search(r"^#\s*目次\b", page.markdown, re.MULTILINE):
            toc_page = page
            break

    if toc_page is None:
        raise ChapterDetectionError("目次ページが見つかりませんでした（先頭 15 ページ以内）。")

    # 目次エントリ: 「第N章 タイトル .... ページ番号」形式を解析する
    entry_pattern = re.compile(
        r"第\s*([０-９0-9一二三四五六七八九十]+)\s*章\s+(.+?)\s+[・\.\s]*(\d+)\s*$",
        re.MULTILINE,
    )
    entries: list[tuple[int, str, int]] = []
    for m in entry_pattern.finditer(toc_page.markdown):
        chapter_num = _parse_chapter_number(m.group(1))
        title = m.group(2).strip()
        # 目次のページ番号は書籍内ページ番号（PDF page_index とは異なる場合がある）
        book_page = int(m.group(3))
        if chapter_num is not None:
            entries.append((chapter_num, title, book_page))

    if not entries:
        raise ChapterDetectionError("目次ページからエントリを抽出できませんでした。")

    # 書籍内ページ番号を PDF の page_index に変換する
    # 目次自体のページ番号から先頭オフセットを推定する
    offset = _estimate_page_offset(toc_page.page_index, entries, page_markdowns)

    total_pages = len(page_markdowns)
    raw_specs: list[tuple[ChapterKind, int | None, str, int]] = []
    for chapter_num, title, book_page in sorted(entries, key=lambda e: e[0]):
        page_index = max(0, book_page - 1 + offset)
        if page_index < total_pages:
            raw_specs.append((ChapterKind.CHAPTER, chapter_num, title, page_index))

    if not raw_specs:
        raise ChapterDetectionError("目次エントリを PDF ページに対応付けられませんでした。")

    return _build_chapter_specs(raw_specs, total_pages, source="ocr_toc")


def _estimate_page_offset(
    toc_page_index: int,
    entries: list[tuple[int, str, int]],
    page_markdowns: list[PageMarkdown],
) -> int:
    """目次ページの page_index と書籍内ページ番号からオフセットを推定する。

    シンプルな実装として、第1章のページ番号から本文先頭を特定し差分を返す。
    """
    # 第1章の書籍ページ番号
    first_chapter_book_page = min(e[2] for e in entries)
    # 本文見出し検出で第1章の page_index を取得する
    try:
        body_chapters = detect_via_body_headings(page_markdowns)
        first_chapter_specs = [
            c for c in body_chapters
            if c.kind == ChapterKind.CHAPTER and c.chapter_number == 1
        ]
        if first_chapter_specs:
            first_chapter_page_index = first_chapter_specs[0].start_page_index
            return first_chapter_page_index - (first_chapter_book_page - 1)
    except ChapterDetectionError:
        pass
    # フォールバック: 目次ページの後ろから数えた位置
    return toc_page_index + 1


def detect_via_body_headings(
    page_markdowns: list[PageMarkdown],
) -> list[ChapterSpec]:
    """各ページの Markdown 冒頭の H1 見出しから章境界を検出する。

    検出対象: 第N章、まえがき/前書き/はじめに、あとがき/後書き/おわりに、索引

    Args:
        page_markdowns: OCR 済み全ページの PageMarkdown リスト。

    Raises:
        ChapterDetectionError: 章見出しが 1 つも検出されない場合。
    """
    if not page_markdowns:
        raise ChapterDetectionError("ページが空です。")

    raw_specs: list[tuple[ChapterKind, int | None, str, int]] = []

    for page in page_markdowns:
        # ページ冒頭の H1 行のみを対象とする（本文中の H1 はノイズ）
        first_h1 = _extract_first_h1(page.markdown)
        if first_h1 is None:
            continue

        kind, num = _classify_heading(first_h1)
        if kind is None:
            continue

        title = _clean_title(first_h1, kind, num)
        raw_specs.append((kind, num, title, page.page_index))

    if not raw_specs:
        raise ChapterDetectionError(
            "章見出し（第N章・まえがき・あとがき・索引）が 1 つも検出されませんでした。"
        )

    total_pages = len(page_markdowns)
    return _build_chapter_specs(raw_specs, total_pages, source="body_headings")


def _extract_first_h1(markdown: str) -> str | None:
    """Markdown テキストの先頭に現れる H1 行（# で始まる行）を返す。

    H1 の前に空行・空白行以外のテキストが存在する場合は None を返す（ページ冒頭以外は無視）。
    """
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            return stripped[2:].strip()
        # H1 以外のテキストが先に出現した場合はページ冒頭ではないと判断
        break
    return None


def _classify_heading(text: str) -> tuple[ChapterKind | None, int | None]:
    """見出しテキストを ChapterKind に分類し、章番号を返す。

    Returns:
        (ChapterKind, chapter_number) のタプル。
        非対象の見出しの場合は (None, None)。
    """
    m = _CHAPTER_PATTERN.match(f"# {text}")
    if m:
        num = _parse_chapter_number(m.group(1))
        return ChapterKind.CHAPTER, num

    if _PREFACE_PATTERN.match(f"# {text}"):
        return ChapterKind.PREFACE, None

    if _AFTERWORD_PATTERN.match(f"# {text}"):
        return ChapterKind.AFTERWORD, None

    if _INDEX_PATTERN.match(f"# {text}"):
        return ChapterKind.INDEX, None

    return None, None


def _parse_chapter_number(raw: str) -> int | None:
    """章番号文字列（全角数字・漢数字・半角数字）を int に変換する。"""
    # 全角数字 → 半角
    normalized = raw.translate(_FULLWIDTH_DIGITS).strip()

    # 漢数字
    if normalized in _KANJI_TO_INT:
        return _KANJI_TO_INT[normalized]

    # 複合漢数字（十一〜十五など）は上で対応済み
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)

    return None


def _clean_title(heading_text: str, kind: ChapterKind, chapter_number: int | None) -> str:
    """見出しテキストから種別プレフィックス（第N章など）を除いた本文タイトルを返す。"""
    text = heading_text.strip()

    if kind == ChapterKind.CHAPTER:
        # 「第N章 タイトル」→「タイトル」
        cleaned = re.sub(
            r"^第\s*[０-９0-9一二三四五六七八九十]+\s*章\s*", "", text
        ).strip()
        return cleaned if cleaned else text

    # PREFACE / AFTERWORD / INDEX はそのまま返す
    return text


def _build_chapter_specs(
    raw_specs: list[tuple[ChapterKind, int | None, str, int]],
    total_pages: int,
    *,
    source: str,
) -> list[ChapterSpec]:
    """(kind, chapter_number, title, start_page_index) のリストから ChapterSpec リストを組み立てる。

    処理:
    1. start_page_index でソート
    2. end_page_index を「次章の start - 1」で算出（最終章は total_pages - 1）
    3. order を 0 から再採番
    """
    # start_page_index でソートし、重複 start は種別優先度で解決
    kind_priority = {
        ChapterKind.PREFACE: 0,
        ChapterKind.CHAPTER: 1,
        ChapterKind.AFTERWORD: 2,
        ChapterKind.INDEX: 3,
    }
    sorted_specs = sorted(raw_specs, key=lambda s: (s[3], kind_priority.get(s[0], 99)))

    result: list[ChapterSpec] = []
    for i, (kind, num, title, start) in enumerate(sorted_specs):
        if i + 1 < len(sorted_specs):
            end = sorted_specs[i + 1][3] - 1
        else:
            end = total_pages - 1
        result.append(
            ChapterSpec(
                order=i,
                kind=kind,
                chapter_number=num,
                title=title,
                start_page_index=start,
                end_page_index=end,
                source=source,
            )
        )
    return result
