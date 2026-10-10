"""仕様: detectors.consistency モジュールのユニットテスト。

書籍内の表記の一貫性（低頻度語・巻末索引の語彙・低頻度の文字 n-gram）から、
地の文の誤字候補を検出する処理と、正解との文字差分による評価を検証する。
"""

from ouj_notebook_converter.detectors.consistency import (
    Candidate,
    body_text,
    detect_index_pages,
    evaluate_page,
    find_index_near_candidates,
    find_rare_ngram_candidates,
    find_rare_word_candidates,
    parse_index_terms,
)


def _candidate(page: int, body: str, surface: str) -> Candidate:
    """body の中の surface を指す候補を作る。"""
    start = body.index(surface)
    return Candidate(
        page=page,
        kind="rare_word",
        surface=surface,
        suggestion="",
        count=1,
        suggestion_count=0,
        start=start,
        end=start + len(surface),
    )


class TestBodyText:
    """body_text: 数式と記法を除いた、日本語の文字だけの地の文を作る。"""

    def test_removes_math_and_markup_and_splits_by_separator(self) -> None:
        """数式・Markdown 記法・句読点は除き、文字の連なりは半角空白で区切る。"""
        markdown = "## 小数の表示\n\n有限小数は $0.25$ のように**終わる**。\n"
        assert body_text(markdown).split() == ["小数の表示", "有限小数は", "のように", "終わる"]


class TestFindRareWordCandidates:
    """find_rare_word_candidates: 低頻度語で、編集距離 1 の高頻度語がある語を候補にする。"""

    def _pages(self) -> dict[int, str]:
        return {
            1: "無制限の議論。" * 5,
            2: "ここに無製限の例がある。",
            3: "整数列と整数の話。" * 5,
        }

    def test_flags_rare_word_with_frequent_neighbor_by_substitution(self) -> None:
        """1 回しか出ない「無製限」は、5 回出る「無制限」と 1 文字違いなので候補になる。"""
        candidates = find_rare_word_candidates(self._pages())
        assert [(c.page, c.surface, c.suggestion) for c in candidates] == [(2, "無製限", "無制限")]
        assert candidates[0].kind == "rare_word"
        assert candidates[0].count == 1
        assert candidates[0].suggestion_count == 5

    def test_offsets_point_to_the_surface_in_body_text(self) -> None:
        """候補の位置は、そのページの body_text 上の文字範囲を指す。"""
        pages = self._pages()
        candidate = find_rare_word_candidates(pages)[0]
        assert body_text(pages[candidate.page])[candidate.start : candidate.end] == "無製限"

    def test_ignores_affix_difference(self) -> None:
        """語頭・語末に 1 文字足しただけの語（整数と整数列）は複合語とみなして候補にしない。"""
        pages = {1: "整数の話。" * 5, 2: "整数列の話。"}
        assert find_rare_word_candidates(pages) == []

    def test_ignores_frequent_neighbor_below_threshold(self) -> None:
        """近い語の出現回数が基準に満たなければ候補にしない。"""
        pages = {1: "無制限の話。" * 2, 2: "無製限の話。"}
        assert find_rare_word_candidates(pages) == []


class TestFindRareNgramCandidates:
    """find_rare_ngram_candidates: かな 1 文字違いの低頻度 n-gram を候補にする。"""

    def test_flags_kana_typo_and_merges_overlapping_grams(self) -> None:
        """「としとう」は 4 回出る「としよう」と 1 文字違いなので、重なる n-gram を 1 件にまとめて候補にする。"""
        pages = {
            1: "情報があるとしよう。" * 4,
            2: "情報があるとしとう。",
        }
        candidates = find_rare_ngram_candidates(pages, n=4, max_rare_count=1, min_neighbor_count=3)
        assert len(candidates) == 1
        assert candidates[0].page == 2
        assert candidates[0].kind == "rare_ngram"
        assert "としとう" in candidates[0].surface


class TestIndexNear:
    """索引の検出・語彙の取り出しと、索引語に近い本文中の語の検出。"""

    _TERMS = (
        "余り",
        "一般角",
        "一般項",
        "因数",
        "因数分解",
        "演算",
        "円周",
        "加法",
        "階差",
        "外角",
    )

    def _index_markdown(self) -> str:
        lines = ["# 索引", "", "### ●あ 行"] + [
            f"{t}　{i + 10}  " for i, t in enumerate(self._TERMS)
        ]
        lines += ["無制限　5  ", "循環小数　59, 60  ", "$e$　257"]
        return "\n".join(lines)

    def test_detects_index_pages_by_line_shape(self) -> None:
        """「用語　ページ番号」の形の行が大半のページを索引とみなす。"""
        pages = {1: "普通の本文です。\n二行目です。", 2: self._index_markdown()}
        assert detect_index_pages(pages) == {2}

    def test_front_table_of_contents_is_not_index(self) -> None:
        """巻頭の目次も同じ形の行が並ぶが、巻末に続く連続したページだけを索引とする。"""
        pages = {
            1: self._index_markdown(),
            2: "普通の本文です。\n二行目です。",
            3: self._index_markdown(),
            4: self._index_markdown(),
        }
        assert detect_index_pages(pages) == {3, 4}

    def test_parse_terms_keeps_only_kanji_katakana_terms(self) -> None:
        """索引語は漢字・カタカナだけの語に限り、数式を含む行は除く。"""
        terms = parse_index_terms(self._index_markdown())
        assert {"無制限", "循環小数"} <= terms
        assert "e" not in terms

    def test_flags_body_word_near_index_term(self) -> None:
        """索引語「無制限」と 1 文字違いの本文中の語「無製限」を候補にする。"""
        pages = {3: "これは無製限に続く。", 4: "循環小数の話。"}
        candidates = find_index_near_candidates(pages, {"無制限", "循環小数"})
        assert [(c.page, c.surface, c.suggestion) for c in candidates] == [(3, "無製限", "無制限")]
        assert candidates[0].kind == "index_near"


class TestEvaluatePage:
    """evaluate_page: 正解との文字差分を真の誤読とみなして候補を採点する。"""

    def test_counts_hits_and_misses_by_overlap_with_diff(self) -> None:
        """誤読箇所に重なる候補は的中、重ならない候補は誤検出、候補の無い誤読は見落としになる。"""
        truth = "著者は隈部正博です。無限に続く。"
        pred = "著者は服部正博です。無制限に続く。"
        pred_body = body_text(pred)
        hit = _candidate(3, pred_body, "服部正博")
        miss = _candidate(3, pred_body, "著者")
        score = evaluate_page(truth, pred, [hit, miss])
        assert (score.flagged, score.true_positive, score.misread, score.detected) == (2, 1, 2, 1)
