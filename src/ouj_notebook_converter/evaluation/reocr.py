"""仕様: 評価セットのページだけを指定の解像度・モデル・プロンプトで再 OCR する実験用ハーネス。

使い方:
  uv run python -m ouj_notebook_converter.evaluation.reocr \\
      --truth <評価セット> --pdf <原本 PDF> --dpi 300 --model gemini-3.8-flash \\
      --out <出力ディレクトリ> [--prompt-file プロンプト.txt] [--run-id run1]

出力（`--run-id` 指定時は `<out>/<run-id>/`）:
  page_NNNN/raw.md     OCR 結果。評価コマンド（`--pred`）にそのまま渡せる
  page_NNNN/usage.json ページごとの処理時間・試行回数・トークン使用量
  run.json             条件（dpi・モデル・プロンプト）と全ページの記録・合計

注意事項:
  - 完了済みページ（raw.md と usage.json が揃っている）は API を呼ばずに読み飛ばす。
    レート制限などで中断した実験を同じコマンドで再開できる。条件を変えるときは出力先も変えること
  - HTTP 429 / 500 / 503 は待って再試行する。それ以外のエラーは即座に失敗する（Fail-Fast）
  - usage_metadata が取得できないレスポンスはコスト比較が成立しないため失敗させる
  - 後処理（normalize_ocr_markdown）は適用しない。評価側が raw.md 読み込み時に適用する
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

import numpy as np
import typer

from ouj_notebook_converter.config import Settings
from ouj_notebook_converter.evaluation.dataset import load_manifest
from ouj_notebook_converter.plugins.ocr.gemini import (
    GeminiAnalyzerResult,
    GeminiUsage,
    create_gemini_analyzer,
)

app = typer.Typer(add_completion=False)

# 再試行の対象とする HTTP ステータス（レート制限・一時的なサーバー側エラー）
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 503})
_DEFAULT_MAX_RETRIES = 6
_DEFAULT_RETRY_WAIT_SECONDS = 10.0
_MAX_RETRY_WAIT_SECONDS = 120.0
_DEFAULT_DPI = 200
_DEFAULT_MODEL = "gemini-3.8-flash"


class _Analyzer(Protocol):
    """GeminiAnalyzer のうち本モジュールが使う部分。"""

    last_usage: GeminiUsage | None

    def __call__(self, image: np.ndarray) -> tuple[GeminiAnalyzerResult, None, None]: ...


@dataclass(frozen=True)
class PageRecord:
    """1 ページ分の再 OCR 記録。

    Attributes:
        page: ページ番号（1 始まり）。
        seconds: 成功した API 呼び出し 1 回の所要時間（秒）。待機や失敗した試行は含まない。
        attempts: API 呼び出しの試行回数（再試行を含む）。
        usage: トークン使用量。
    """

    page: int
    seconds: float
    attempts: int
    usage: GeminiUsage | None

    def to_dict(self) -> dict[str, Any]:
        usage = self.usage
        return {
            "page": self.page,
            "seconds": self.seconds,
            "attempts": self.attempts,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "thinking_tokens": usage.thinking_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageRecord:
        return cls(
            page=data["page"],
            seconds=data["seconds"],
            attempts=data["attempts"],
            usage=GeminiUsage(
                prompt_tokens=data["prompt_tokens"],
                output_tokens=data["output_tokens"],
                thinking_tokens=data["thinking_tokens"],
                total_tokens=data["total_tokens"],
            ),
        )


def resolve_output_dir(out: Path, run_id: str | None) -> Path:
    """出力ディレクトリを決める。run_id があれば `out` 配下のサブディレクトリにする。

    Raises:
        ValueError: run_id にパス区切りや `..` が含まれる場合。
    """
    if run_id is None:
        return out
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError(f"--run-id に使えない文字列です（パス区切りは不可）: {run_id!r}")
    return out / run_id


def render_pdf_page(pdf_path: Path, page: int, dpi: int) -> np.ndarray:
    """PDF の 1 ページ（1 始まり）を BGR の uint8 ndarray にレンダリングする。

    ounc 本体（pipeline/stages/load_pypdfium.py）と同じ pypdfium2 のレンダリング条件を使う。
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page <= len(doc):
            raise ValueError(f"ページ {page} は PDF の範囲外です（全 {len(doc)} ページ）")
        bitmap = doc[page - 1].render(scale=dpi / 72.0)
        return np.copy(bitmap.to_numpy())
    finally:
        doc.close()


def _status_code(error: RuntimeError) -> int | None:
    """GeminiAnalyzer が包んだ元例外から HTTP ステータスコードを取り出す。"""
    code = getattr(error.__cause__, "code", None)
    return code if isinstance(code, int) else None


def run_reocr(
    *,
    pages: list[int],
    render_page: Callable[[int], np.ndarray],
    analyzer: _Analyzer,
    out_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_wait_seconds: float = _DEFAULT_RETRY_WAIT_SECONDS,
) -> list[PageRecord]:
    """指定ページを再 OCR し、`out_dir/page_NNNN/` に結果と記録を書き出す。

    Args:
        pages: 再 OCR するページ番号（1 始まり）。
        render_page: ページ番号から BGR 画像を返す関数。
        analyzer: Gemini アナライザー（呼び出し後に last_usage を持つこと）。
        out_dir: 出力ディレクトリ。
        sleep: 再試行の待機関数（テストで差し替える）。
        clock: 経過時間の計測関数（テストで差し替える）。
        max_retries: 再試行の最大回数（初回を除く）。
        retry_wait_seconds: 最初の待機秒数。再試行ごとに 2 倍にし、上限は 120 秒。

    Returns:
        pages と同じ順序の記録。

    Raises:
        RuntimeError: 再試行対象外の API エラー、再試行回数の超過、usage_metadata の欠落。
    """
    records: list[PageRecord] = []
    for page in pages:
        page_dir = out_dir / f"page_{page:04d}"
        raw_path = page_dir / "raw.md"
        usage_path = page_dir / "usage.json"
        if raw_path.is_file() and usage_path.is_file():
            records.append(PageRecord.from_dict(json.loads(usage_path.read_text(encoding="utf-8"))))
            continue

        image = render_page(page)
        wait = retry_wait_seconds
        attempts = 0
        while True:
            attempts += 1
            started = clock()
            try:
                result, _, _ = analyzer(image)
                seconds = clock() - started
                break
            except RuntimeError as e:
                if _status_code(e) not in _RETRYABLE_STATUS_CODES:
                    raise
                if attempts > max_retries:
                    raise RuntimeError(
                        f"p.{page}: 再試行 {max_retries} 回でも成功しませんでした: {e.__cause__}"
                    ) from e
                typer.echo(
                    f"p.{page}: {e.__cause__} のため {wait:.0f} 秒待って再試行します", err=True
                )
                sleep(wait)
                wait = min(wait * 2, _MAX_RETRY_WAIT_SECONDS)

        usage = analyzer.last_usage
        if usage is None:
            raise RuntimeError(f"p.{page}: レスポンスに usage_metadata がありません")
        record = PageRecord(page=page, seconds=seconds, attempts=attempts, usage=usage)
        page_dir.mkdir(parents=True, exist_ok=True)
        result.to_markdown(raw_path)
        # usage.json を最後に書くことで、途中で中断したページを完了済みと誤認しない
        usage_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
    return records


def _sum_tokens(records: list[PageRecord], field: str) -> int:
    return sum(getattr(r.usage, field) or 0 for r in records if r.usage is not None)


def _build_summary(
    records: list[PageRecord],
    *,
    dpi: int,
    model: str,
    run_id: str | None,
    prompt_file: Path | None,
) -> dict[str, Any]:
    """条件・ページ別記録・合計をまとめた run.json の内容を作る。"""
    return {
        "dpi": dpi,
        "model": model,
        "run_id": run_id,
        "prompt_file": str(prompt_file) if prompt_file else None,
        "pages": [r.to_dict() for r in records],
        "totals": {
            "pages": len(records),
            "seconds": sum(r.seconds for r in records),
            "prompt_tokens": _sum_tokens(records, "prompt_tokens"),
            "output_tokens": _sum_tokens(records, "output_tokens"),
            "thinking_tokens": _sum_tokens(records, "thinking_tokens"),
            "total_tokens": _sum_tokens(records, "total_tokens"),
        },
    }


@app.command()
def reocr(
    truth: Annotated[
        Path, typer.Option("--truth", help="評価セット（manifest.json のページを対象にする）")
    ],
    pdf: Annotated[Path, typer.Option("--pdf", help="原本 PDF")],
    out: Annotated[Path, typer.Option("--out", help="出力ディレクトリ")],
    dpi: Annotated[int, typer.Option("--dpi", help="レンダリング解像度")] = _DEFAULT_DPI,
    model: Annotated[str, typer.Option("--model", help="Gemini モデル名")] = _DEFAULT_MODEL,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", help="OCR プロンプトのテキストファイル（省略時は既定）"),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="繰り返し実行の識別子（<out>/<run-id>/ に出力）")
    ] = None,
    api_key: Annotated[
        str | None, typer.Option("--api-key", envvar="GEMINI_API_KEY", help="Gemini API キー")
    ] = None,
) -> None:
    """評価セットのページだけを指定条件で再 OCR し、評価コマンドに渡せる形式で出力する。"""
    try:
        api_key = api_key or Settings().gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("--api-key または GEMINI_API_KEY 環境変数が必要です")
        if not pdf.is_file():
            raise FileNotFoundError(f"原本 PDF が見つかりません: {pdf}")
        out_dir = resolve_output_dir(out, run_id)
        pages = [entry.page for entry in load_manifest(truth)]
        analyzer_kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
        if prompt_file is not None:
            analyzer_kwargs["prompt"] = prompt_file.read_text(encoding="utf-8")
        analyzer = create_gemini_analyzer(**analyzer_kwargs)

        out_dir.mkdir(parents=True, exist_ok=True)
        records = run_reocr(
            pages=pages,
            render_page=lambda page: render_pdf_page(pdf, page, dpi),
            analyzer=analyzer,
            out_dir=out_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

    summary = _build_summary(records, dpi=dpi, model=model, run_id=run_id, prompt_file=prompt_file)
    (out_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    totals = summary["totals"]
    typer.echo(
        f"{totals['pages']} ページを再 OCR しました: {out_dir}"
        f"（{totals['seconds']:.0f} 秒、入力 {totals['prompt_tokens']} / "
        f"出力 {totals['output_tokens']} / 思考 {totals['thinking_tokens']} トークン）"
    )


if __name__ == "__main__":
    app()
