"""仕様: judge.locate のユニットテスト。

Gemini 出力には bbox が無いため、行を yomitoku の段落に対応付けて原本画像上の位置を推定する。
日本語を含む行は段落に直接対応付け、数式だけの行は前後の対応付けできた段落に挟まれた帯とする。
"""

from ouj_notebook_converter.detectors.cross_ocr_diff.detect import parse_analysis
from ouj_notebook_converter.judge.locate import locate_candidate, locate_line
from ouj_notebook_converter.judge.models import JudgeCandidate

_UPPER = "有理数を小数で表すと、有限小数か循環小数になります。"
_LOWER = "したがって、この数は有理数であることが分かります。"
_RAW = f"{_UPPER}\n\n$$(\\sqrt[3]{{a}})^3 = (\\sqrt[3]{{a}})^4$$\n\n{_LOWER}\n"


def _analysis() -> object:
    return parse_analysis(
        {
            "paragraphs": [
                {"box": [100, 100, 900, 160], "contents": _UPPER, "role": None},
                {"box": [300, 200, 700, 260], "contents": "(スqrt3a)3=(3a)4", "role": None},
                {"box": [100, 300, 900, 360], "contents": _LOWER, "role": None},
            ],
            "figures": [],
            "tables": [],
            "words": [],
        },
        source="テスト",
    )


def test_日本語を含む行は対応する段落の範囲になる() -> None:
    result = locate_line(_RAW.split("\n"), 0, _analysis())  # type: ignore[arg-type]
    assert result == ((100, 100, 900, 160), "line_alignment")


def test_数式だけの行は前後の段落に挟まれた帯になる() -> None:
    result = locate_line(_RAW.split("\n"), 2, _analysis())  # type: ignore[arg-type]
    assert result is not None
    bbox, method = result
    assert method == "band_between"
    # 上の段落の下端から下の段落の上端までを含み、横幅は前後の段落の和集合になる
    assert bbox[1] <= 160 and bbox[3] >= 300
    assert bbox[0] == 100 and bbox[2] == 900


def test_前の段落しか無い数式行は下方向に一定の高さを取る() -> None:
    raw = f"{_UPPER}\n\n$$a = b$$\n"
    result = locate_line(raw.split("\n"), 2, _analysis())  # type: ignore[arg-type]
    assert result is not None
    bbox, method = result
    assert method == "band_between"
    assert bbox[1] == 160 and bbox[3] > 160


def test_対応付けできる行が1つも無ければNoneを返す() -> None:
    assert locate_line(["$$a = b$$"], 0, _analysis()) is None  # type: ignore[arg-type]


def test_段落と似ていない行は対応付けない() -> None:
    raw = "全く別の内容の文章がここに書かれています。\n"
    assert locate_line(raw.split("\n"), 0, _analysis()) is None  # type: ignore[arg-type]


def _candidate(**overrides: object) -> JudgeCandidate:
    base = {
        "id": "p0001-equation_check-1",
        "page": 1,
        "source": "equation_check",
        "kind": "violated",
        "text": "a = b",
        "hint": "",
        "excerpt": "",
        "excerpt_scope": "lines",
        "line_index": 2,
    }
    return JudgeCandidate(**{**base, **overrides})  # type: ignore[arg-type]


def test_locate_candidate_は既に_bbox_がある候補を変えない() -> None:
    candidate = _candidate(bbox=(1, 2, 3, 4), location="yomitoku_word")
    assert locate_candidate(candidate, _RAW, _analysis()) is candidate  # type: ignore[arg-type]


def test_locate_candidate_は_bbox_が行の対応先と一致すれば変えない() -> None:
    """yomitoku の単語の bbox が、Gemini の行を対応付けた段落の中にあれば、その精密な位置を残す。"""
    candidate = _candidate(line_index=0, bbox=(200, 110, 300, 150), location="yomitoku_word")
    assert locate_candidate(candidate, _RAW, _analysis()) is candidate  # type: ignore[arg-type]


def test_locate_candidate_は_bbox_が行の対応先とずれていれば対応先に置き換える() -> None:
    """yomitoku 側の対応の取り違えで bbox が別の段落を指すと、LLM は別の箇所の画像を見て誤判定する。"""
    candidate = _candidate(line_index=0, bbox=(200, 310, 300, 350), location="yomitoku_word")
    located = locate_candidate(candidate, _RAW, _analysis())  # type: ignore[arg-type]
    assert located.bbox == (100, 100, 900, 160)
    assert located.location == "line_alignment"


def test_locate_candidate_は推定した位置を書き込む() -> None:
    located = locate_candidate(_candidate(), _RAW, _analysis())  # type: ignore[arg-type]
    assert located.location == "band_between"
    assert located.bbox is not None


def test_locate_candidate_は特定できなければページ全体のまま() -> None:
    candidate = _candidate(line_index=None, excerpt_scope="page")
    located = locate_candidate(candidate, _RAW, _analysis())  # type: ignore[arg-type]
    assert located.bbox is None
    assert located.location == "page"
