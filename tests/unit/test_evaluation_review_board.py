"""仕様: evaluation.review_board モジュール（校正台の生成）のユニットテスト。

KaTeX アセットの読み込み、ページ画像の生成、data.json の組み立て、index.html の組み立てを
テストする。KaTeX の dist は偽のディレクトリを作って差し替える。
"""

import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

from ouj_notebook_converter.evaluation import review_board
from ouj_notebook_converter.evaluation.review_board import (
    KatexAssetsError,
    app,
    build_board_data,
    build_review_board,
    load_katex_assets,
    render_index_html,
    render_page_images,
)
from ouj_notebook_converter.pipeline.stages.markdown_cleanup import MATH_SPAN


def _make_katex_dist(root: Path) -> Path:
    dist = root / "dist"
    (dist / "fonts").mkdir(parents=True)
    (dist / "fonts" / "KaTeX_Main-Regular.woff2").write_bytes(b"FONTDATA")
    (dist / "katex.min.css").write_text(
        "@font-face{font-family:KaTeX_Main;src:url(fonts/KaTeX_Main-Regular.woff2) "
        "format('woff2'),url(fonts/KaTeX_Main-Regular.woff) format('woff'),"
        "url(fonts/KaTeX_Main-Regular.ttf) format('truetype')}.katex{font:normal 1.21em KaTeX_Main}",
        encoding="utf-8",
    )
    (dist / "katex.min.js").write_text("window.katex={};", encoding="utf-8")
    return dist


def _make_truth(root: Path) -> Path:
    truth = root / "eval"
    truth.mkdir(parents=True)
    (truth / "manifest.json").write_text(
        json.dumps(
            {"pages": [{"page": 2, "category": "数式中心"}, {"page": 1, "category": "地の文中心"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (truth / "page_0001.md").write_text("# 第1章\n\n本文です。", encoding="utf-8")
    (truth / "page_0002.md").write_text("$x^2 + 1$ を考える。", encoding="utf-8")
    return truth


def _make_pdf(path: Path, page_count: int = 2) -> Path:
    c = canvas.Canvas(str(path), pagesize=(200, 300))
    for i in range(page_count):
        c.drawString(20, 150, f"page {i + 1}")
        c.showPage()
    c.save()
    return path


class TestLoadKatexAssets:
    def test_CSSのフォント参照をwoff2のdataURIに置き換える(self, tmp_path: Path) -> None:
        css, js = load_katex_assets(_make_katex_dist(tmp_path))

        assert "data:font/woff2;base64," in css
        assert "fonts/" not in css
        assert ".woff)" not in css
        assert ".ttf)" not in css
        assert js == "window.katex={};"

    def test_distが無ければnpm_ciを案内して失敗する(self, tmp_path: Path) -> None:
        with pytest.raises(KatexAssetsError, match="npm ci"):
            load_katex_assets(tmp_path / "missing")

    def test_フォントファイルが欠けていれば失敗する(self, tmp_path: Path) -> None:
        dist = _make_katex_dist(tmp_path)
        (dist / "fonts" / "KaTeX_Main-Regular.woff2").unlink()

        with pytest.raises(KatexAssetsError, match=r"KaTeX_Main-Regular\.woff2"):
            load_katex_assets(dist)


class TestRenderPageImages:
    def test_指定ページをグレースケールJPEGで書き出す(self, tmp_path: Path) -> None:
        from PIL import Image

        pdf = _make_pdf(tmp_path / "book.pdf")
        paths = render_page_images(pdf, [2], tmp_path / "img", dpi=72)

        assert list(paths) == [2]
        assert paths[2] == tmp_path / "img" / "page_0002.jpg"
        with Image.open(paths[2]) as image:
            assert image.format == "JPEG"
            assert image.mode == "L"
            assert image.size == (200, 300)

    def test_PDFの範囲外のページは失敗する(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "book.pdf", page_count=1)

        with pytest.raises(ValueError, match="ページ 5"):
            render_page_images(pdf, [5], tmp_path / "img", dpi=72)


class TestBuildBoardData:
    def test_正解のみならmanifest順にページを並べる(self, tmp_path: Path) -> None:
        data = build_board_data(_make_truth(tmp_path), None, title="試験")

        assert data["title"] == "試験"
        assert [p["page"] for p in data["pages"]] == [2, 1]
        assert data["pages"][0]["truth"] == "$x^2 + 1$ を考える。"
        assert data["pages"][0]["pred"] is None
        assert data["pages"][0]["image"] == "img/page_0002.jpg"
        assert data["pages"][0]["category"] == "数式中心"
        assert data["hasPred"] is False

    def test_boardIdは評価セットの内容で決まり再生成では変わらない(self, tmp_path: Path) -> None:
        truth_a = _make_truth(tmp_path / "a")
        truth_b = _make_truth(tmp_path / "b")
        (truth_b / "manifest.json").write_text(
            json.dumps({"pages": [{"page": 7, "category": "筆算・表"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (truth_b / "page_0007.md").write_text("筆算の例", encoding="utf-8")

        first = build_board_data(truth_a, None, title="校正台")["boardId"]
        again = build_board_data(truth_a, None, title="別の表題")["boardId"]
        other = build_board_data(truth_b, None, title="校正台")["boardId"]

        assert first == again
        assert first != other

    def test_数式の検出規則はMATH_SPANと同じ正規表現を渡す(self, tmp_path: Path) -> None:
        data = build_board_data(_make_truth(tmp_path), None, title="試験")

        assert data["mathSpanPattern"] == MATH_SPAN.pattern

    def test_比較対象と照合記録を読み込む(self, tmp_path: Path) -> None:
        truth = _make_truth(tmp_path)
        (truth / "review").mkdir()
        (truth / "review" / "page_0001.md").write_text("要確認なし", encoding="utf-8")
        pred = tmp_path / "pred"
        (pred / "page_0001").mkdir(parents=True)
        (pred / "page_0001" / "raw.md").write_text("# 第一章\n\n本文です。", encoding="utf-8")
        (pred / "page_0002.md").write_text("$x^2 - 1$ を考える。", encoding="utf-8")

        data = build_board_data(truth, pred, title="試験", pred_label="下書き")

        by_page = {p["page"]: p for p in data["pages"]}
        assert by_page[2]["pred"] == "$x^2 - 1$ を考える。"
        assert by_page[1]["pred"].startswith("# 第一章")
        assert by_page[1]["notes"] == "要確認なし"
        assert by_page[2]["notes"] is None
        assert data["hasPred"] is True
        assert data["predLabel"] == "下書き"

    def test_比較対象にページが無ければ失敗する(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred"
        pred.mkdir()

        with pytest.raises(FileNotFoundError):
            build_board_data(_make_truth(tmp_path), pred, title="試験")


class TestRenderIndexHtml:
    def test_プレースホルダを埋め込み終了タグをエスケープする(self) -> None:
        template = (
            "<style>/*__KATEX_CSS__*/</style><script>/*__KATEX_JS__*/</script>"
            "<script>__DATA_JSON__</script>"
        )
        html = render_index_html(template, {"memo": "</script>"}, "CSS本体", "JS本体")

        assert "CSS本体" in html
        assert "JS本体" in html
        assert "__DATA_JSON__" not in html
        # データ中の </script> が本物の終了タグとして残っていない（開始タグ 2 つ・終了タグ 2 つ）
        assert html.count("</script>") == 2

    def test_プレースホルダが無いテンプレートは失敗する(self) -> None:
        with pytest.raises(ValueError, match="プレースホルダ"):
            render_index_html("<html></html>", {}, "", "")


class TestTemplate:
    def test_プレースホルダは各1か所だけ含まれる(self) -> None:
        template = (
            Path(review_board.__file__).parent / "templates" / "review_board.html"
        ).read_text(encoding="utf-8")

        for placeholder in ("/*__KATEX_CSS__*/", "/*__KATEX_JS__*/", "__DATA_JSON__"):
            assert template.count(placeholder) == 1

    def test_KaTeXのバージョンは検査スクリプトと同じ(self) -> None:
        package_json = (
            Path(review_board.__file__).parents[3] / "scripts" / "katex_check" / "package.json"
        )

        version = json.loads(package_json.read_text(encoding="utf-8"))["dependencies"]["katex"]
        assert review_board.KATEX_VERSION == version


class TestBuildReviewBoard:
    def test_index_data_img一式を出力する(self, tmp_path: Path) -> None:
        truth = _make_truth(tmp_path)
        pdf = _make_pdf(tmp_path / "book.pdf")
        out = tmp_path / "out"

        build_review_board(
            truth, pdf, out, pred_dir=None, katex_dist=_make_katex_dist(tmp_path), title="試験"
        )

        assert (out / "img" / "page_0001.jpg").is_file()
        assert (out / "img" / "page_0002.jpg").is_file()
        data = json.loads((out / "data.json").read_text(encoding="utf-8"))
        assert data["katexVersion"] == "0.18.9"
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "window.katex={};" in html
        assert "data:font/woff2;base64," in html
        assert "第1章" in html
        assert "reviews" in html


class TestCli:
    def test_コマンドで一式を生成する(self, tmp_path: Path) -> None:
        truth = _make_truth(tmp_path)
        pdf = _make_pdf(tmp_path / "book.pdf")
        dist = _make_katex_dist(tmp_path)
        out = tmp_path / "out"

        result = CliRunner().invoke(
            app,
            [
                "--truth", str(truth),
                "--pdf", str(pdf),
                "--out", str(out),
                "--katex-dist", str(dist),
                "--dpi", "72",
            ],
        )  # fmt: skip

        assert result.exit_code == 0, result.output
        assert (out / "index.html").is_file()

    def test_KaTeXが無ければ終了コード1で案内する(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--truth", str(_make_truth(tmp_path)),
                "--pdf", str(_make_pdf(tmp_path / "b.pdf")),
                "--out", str(tmp_path / "out"),
                "--katex-dist", str(tmp_path / "none"),
            ],
        )  # fmt: skip

        assert result.exit_code == 1
        assert "npm ci" in result.output
