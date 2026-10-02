"""仕様: 評価セットと評価対象の出力をページ単位で読み込む。

評価セットのディレクトリ構成:
  manifest.json   {"pages": [{"page": 70, "category": "数式中心"}, ...]}
  page_0070.md    正解 Markdown（ページ番号は 1 始まり、4 桁ゼロ埋め）

評価対象のディレクトリは次のどちらかの構成を受け付ける:
  - `ounc --no-combine` の出力: page_0070.md
  - ページキャッシュ: page_0070/raw.md（変換時と同じ後処理 normalize_ocr_markdown を適用して読む）

注意事項:
  - キャッシュからの読み込みでは数式 overlay（--math-backend pix2text）は再現しない
  - ページが欠けている・構成が曖昧な場合は推測せず例外を送出する（Fail-Fast）
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ouj_notebook_converter.pipeline.stages.markdown_cleanup import normalize_ocr_markdown

MANIFEST_FILENAME = "manifest.json"


class EvalPage(BaseModel):
    """評価セットに含まれる 1 ページ。

    Attributes:
        page: ページ番号（1 始まり）。
        category: ページの層（「地の文中心」「数式中心」など）。層別集計に使う。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    page: int = Field(gt=0)
    category: str = Field(min_length=1)


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: list[EvalPage]


def load_manifest(truth_dir: Path) -> list[EvalPage]:
    """評価セットの manifest.json を読み込み、評価ページを記載順に返す。

    Raises:
        FileNotFoundError: manifest.json が存在しない場合。
        ValueError: 形式が不正、またはページ番号が重複している場合。
    """
    manifest_path = truth_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"評価セットの {MANIFEST_FILENAME} が見つかりません: {manifest_path}"
        )
    try:
        manifest = _Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"{manifest_path} の形式が不正です: {e}") from e

    seen: set[int] = set()
    for entry in manifest.pages:
        if entry.page in seen:
            raise ValueError(f"{manifest_path} でページ {entry.page} が重複しています")
        seen.add(entry.page)
    return manifest.pages


def read_truth(truth_dir: Path, page: int) -> str:
    """正解 Markdown（page_NNNN.md）を読み込む。

    Raises:
        FileNotFoundError: 正解ファイルが存在しない場合。
    """
    path = truth_dir / f"{_page_stem(page)}.md"
    if not path.is_file():
        raise FileNotFoundError(f"正解ファイルが見つかりません: {path}")
    return path.read_text(encoding="utf-8")


def read_prediction(pred_dir: Path, page: int) -> str:
    """評価対象のページ Markdown を読み込む。

    Raises:
        FileNotFoundError: どちらの構成のファイルも存在しない場合。
        ValueError: 両方の構成のファイルが存在し、どちらを評価すべきか決められない場合。
    """
    output_path = pred_dir / f"{_page_stem(page)}.md"
    cache_path = pred_dir / _page_stem(page) / "raw.md"
    has_output = output_path.is_file()
    has_cache = cache_path.is_file()
    if has_output and has_cache:
        raise ValueError(
            f"{output_path} と {cache_path} の両方が存在するため、評価対象を決められません"
        )
    if has_output:
        return output_path.read_text(encoding="utf-8")
    if has_cache:
        return normalize_ocr_markdown(cache_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"評価対象のページが見つかりません: {output_path} または {cache_path}")


def _page_stem(page: int) -> str:
    return f"page_{page:04d}"
