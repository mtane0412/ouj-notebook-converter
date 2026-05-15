"""仕様: tiny_2page.pdf を使ったエンドツーエンド統合テスト。

Yomitoku を実際に使用するため @pytest.mark.slow 付き。
デフォルトでは pytest -m "not slow" でスキップされる。
yomitoku と ocr extra が未インストールの場合は自動的にスキップする。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# yomitoku が使えるかチェック
try:
    import yomitoku  # type: ignore[import-untyped]  # noqa: F401
    YOMITOKU_AVAILABLE = True
except ImportError:
    YOMITOKU_AVAILABLE = False


@pytest.fixture(scope="session")
def tiny_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """テスト用の 2 ページ PDF を reportlab で生成する。"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab が未インストール。統合テストをスキップします。")

    out_dir = tmp_path_factory.mktemp("fixtures")
    pdf_path = out_dir / "tiny_2page.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    _, height = A4

    # ページ 1
    c.setFont("Helvetica", 24)
    c.drawString(72, height - 100, "Chapter 1: Introduction")
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 150, "This is the first page of the test document.")
    c.drawString(72, height - 170, "It contains sample text for OCR testing.")
    c.showPage()

    # ページ 2
    c.setFont("Helvetica", 24)
    c.drawString(72, height - 100, "Chapter 2: Analysis")
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 150, "This is the second page of the test document.")
    c.drawString(72, height - 170, "Data analysis and knowledge discovery.")
    c.showPage()

    c.save()
    return pdf_path


@pytest.mark.slow
@pytest.mark.skipif(not YOMITOKU_AVAILABLE, reason="yomitoku が未インストール (uv sync --extra ocr)")
class TestEndToEndTiny:
    """tiny_2page.pdf を使ったエンドツーエンドテスト。"""

    def test_convert_コマンドが正常終了する(self, tiny_pdf: Path, tmp_path: Path) -> None:
        """ounc convert コマンドが exit_code=0 で終了することを確認。"""
        result = subprocess.run(
            [
                sys.executable, "-m", "ouj_notebook_converter",
                "convert", str(tiny_pdf),
                "--outdir", str(tmp_path / "out"),
                "--device", "cpu",
                "--lite",
                "--pages", "1-2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"convert コマンドが失敗しました。\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_Markdownファイルが生成される(self, tiny_pdf: Path, tmp_path: Path) -> None:
        """出力先に .md ファイルが作成されることを確認。"""
        out_dir = tmp_path / "out_md"
        subprocess.run(
            [
                sys.executable, "-m", "ouj_notebook_converter",
                "convert", str(tiny_pdf),
                "--outdir", str(out_dir),
                "--device", "cpu",
                "--lite",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 1, f"Markdown ファイルが見つかりません: {list(out_dir.iterdir())}"
        content = md_files[0].read_text(encoding="utf-8")
        assert len(content) > 10, "Markdown の内容が空に近いです"

    def test_ページマーカーが含まれる(self, tiny_pdf: Path, tmp_path: Path) -> None:
        """出力 Markdown にページマーカーが含まれることを確認。"""
        out_dir = tmp_path / "out_marker"
        subprocess.run(
            [
                sys.executable, "-m", "ouj_notebook_converter",
                "convert", str(tiny_pdf),
                "--outdir", str(out_dir),
                "--device", "cpu",
                "--lite",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        md_file = next(out_dir.glob("*.md"))
        content = md_file.read_text(encoding="utf-8")
        assert "<!-- page:" in content, "ページマーカーが見つかりません"

    def test_2回目実行はキャッシュから再現できる(self, tiny_pdf: Path, tmp_path: Path) -> None:
        """同じコマンドを 2 回実行しても正常終了することを確認（M2 でキャッシュ高速化）。"""
        out_dir = tmp_path / "out_cache"
        cmd = [
            sys.executable, "-m", "ouj_notebook_converter",
            "convert", str(tiny_pdf),
            "--outdir", str(out_dir),
            "--device", "cpu",
            "--lite",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result2.returncode == 0
