"""仕様: analysis.json から数式 paragraph を抽出し、LaTeX に変換する math_extract ステージ。

処理フロー:
  1. analysis.json を読み込み、role が inline_formula / display_formula の paragraph を抽出
  2. 元画像から bbox をクロップして cache_page_dir/math/ に PNG として保存
  3. MathEngineProtocol を通じて LaTeX 文字列に変換
  4. MathOverlay（クロップ画像 → LaTeX の対応）を返す
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# yomitoku の save_image を再利用（cv2 依存の追加を避ける）
from yomitoku.utils.misc import save_image

from ouj_notebook_converter.pipeline.types import MathOverlay, PageAnalysis
from ouj_notebook_converter.plugins.math.base import MathEngineProtocol

_MATH_ROLES = frozenset({"inline_formula", "display_formula"})


@dataclass(frozen=True)
class MathParagraph:
    """analysis.json から抽出した数式 paragraph の情報。"""

    index: int
    role: str
    box: tuple[int, int, int, int]
    original_contents: str


def extract_math_paragraphs(analysis_json_path: Path) -> list[MathParagraph]:
    """analysis.json から数式 role の paragraph を抽出する純粋関数。

    Args:
        analysis_json_path: yomitoku が出力した analysis.json のパス。

    Returns:
        数式 paragraph のリスト（元のリストインデックスと role を保持）。
        contents が None のものは除外する。
    """
    data = json.loads(analysis_json_path.read_text(encoding="utf-8"))
    result: list[MathParagraph] = []
    for idx, p in enumerate(data["paragraphs"]):
        role = p.get("role")
        if role not in _MATH_ROLES:
            continue
        contents = p.get("contents")
        if contents is None:
            continue
        box_raw = p["box"]
        box = (int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3]))
        result.append(MathParagraph(index=idx, role=role, box=box, original_contents=contents))
    return result


def crop_math_image(
    image: np.ndarray,
    paragraph: MathParagraph,
    output_dir: Path,
) -> Path:
    """元画像から数式 paragraph の bbox をクロップして PNG として保存する。

    Args:
        image: BGR 形式の NumPy 配列（yomitoku の load_pdf が返す形式）。
        paragraph: 抽出済みの数式 paragraph 情報。
        output_dir: クロップ画像の保存先ディレクトリ（自動作成）。

    Returns:
        保存した PNG ファイルのパス（output_dir / f"{paragraph.index:04d}.png"）。

    Raises:
        ValueError: クランプ後の bbox サイズが 0 になった場合。
    """
    x1, y1, x2, y2 = paragraph.box
    h, w = image.shape[:2]

    # bbox を画像範囲にクランプ（最小値保証なし、0チェックは後で行う）
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    # クランプ後のサイズが 0 になった場合は Fail-Fast
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"数式 bbox のサイズが 0 です: box={paragraph.box}, image_size=({w}x{h})")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paragraph.index:04d}.png"

    crop = image[y1:y2, x1:x2, :]
    save_image(crop, str(output_path))

    return output_path


def math_extract(
    image: np.ndarray,
    analysis: PageAnalysis,
    cache_page_dir: Path,
    *,
    engine: MathEngineProtocol,
) -> MathOverlay:
    """1 ページ分の数式抽出ステージ。

    analysis.json から数式 paragraph を抽出し、元画像からクロップした PNG を
    engine で LaTeX に変換して MathOverlay として返す。

    Args:
        image: BGR ndarray（ページ全体の画像）。
        analysis: analyze ステージの出力（yomitoku_json_path を使用）。
        cache_page_dir: ページキャッシュディレクトリ（math/ サブディレクトリを作成）。
        engine: MathEngineProtocol を満たす数式エンジン。

    Returns:
        MathOverlay（数式なしの場合は全フィールドが空）。
    """
    paragraphs = extract_math_paragraphs(analysis.yomitoku_json_path)
    if not paragraphs:
        return MathOverlay()

    math_dir = cache_page_dir / "math"
    items: dict[Path, str] = {}
    roles: dict[Path, str] = {}
    originals: dict[Path, str] = {}

    for paragraph in paragraphs:
        crop_path = crop_math_image(image, paragraph, math_dir)
        latex = engine.recognize(crop_path)
        items[crop_path] = latex
        roles[crop_path] = paragraph.role
        originals[crop_path] = paragraph.original_contents

    return MathOverlay(items=items, roles=roles, originals=originals)
