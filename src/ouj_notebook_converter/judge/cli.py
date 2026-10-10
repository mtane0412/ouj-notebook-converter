"""仕様: 誤読候補の LLM 判定（issue #24）の CLI。

使い方（モジュール単位で実行する）:
  # 1. 各検出器の候補を集め、原本画像上の位置を推定する
  uv run python -m ouj_notebook_converter.judge collect \\
      --pred <Gemini のキャッシュ> --yomitoku <yomitoku のキャッシュ> --out candidates.json
  # 2. 候補を原本画像と一緒に Gemini に判定させる（GEMINI_API_KEY が必要）
  uv run python -m ouj_notebook_converter.judge judge \\
      --candidates candidates.json --pdf <原本.pdf> --model gemini-3.1-pro-preview --out judgments.json
  # 3. 判定結果を修正ファイル（#28 の形式）に書き出す
  uv run python -m ouj_notebook_converter.judge export \\
      --candidates candidates.json --judgments judgments.json --pred <Gemini のキャッシュ> --out-dir out
  # 4. 評価セットで判定の正誤と指標の改善幅を測る
  uv run python -m ouj_notebook_converter.judge evaluate ...

エラーは終了コード 1 で停止する（Fail-Fast）。
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ouj_notebook_converter.corrections import CorrectionError, load_corrections
from ouj_notebook_converter.detectors.cross_ocr_diff.detect import parse_analysis
from ouj_notebook_converter.evaluation.dataset import load_manifest
from ouj_notebook_converter.evaluation.katex import KatexCheckError, check_katex
from ouj_notebook_converter.judge.candidates import collect_candidates, read_raw_pages
from ouj_notebook_converter.judge.evaluate import accuracy_report, derive_labels, measure_scores
from ouj_notebook_converter.judge.export import build_exports, write_exports
from ouj_notebook_converter.judge.images import (
    PageRenderer,
    crop_region,
    encode_jpeg,
    fit_long_side,
)
from ouj_notebook_converter.judge.judge import GeminiLlmClient, ImageKind, judge_all
from ouj_notebook_converter.judge.locate import locate_candidate
from ouj_notebook_converter.judge.models import (
    ALL_SOURCES,
    JudgeCandidate,
    Judgment,
    Usage,
    load_candidates,
    save_candidates,
)
from ouj_notebook_converter.judge.policy import DEFAULT_MAX_EDIT_CHARS, AutoPolicy, combine
from ouj_notebook_converter.judge.pricing import estimate_cost

app = typer.Typer(add_completion=False, no_args_is_help=True)

JUDGMENTS_FORMAT_VERSION = 1
# ページ全体を渡すときの長辺の上限（px）。200 DPI の描画は長辺 1650 px 前後
DEFAULT_PAGE_MAX_SIDE = 1200
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_WORKERS = 4


def _fail(message: str) -> typer.Exit:
    typer.echo(f"エラー: {message}", err=True)
    return typer.Exit(code=1)


@app.command()
def collect(
    pred: Annotated[
        Path, typer.Option("--pred", help="Gemini の出力（--no-combine）またはキャッシュ")
    ],
    out: Annotated[Path, typer.Option("--out", help="候補を書き出す JSON ファイル")],
    yomitoku: Annotated[
        Path | None,
        typer.Option("--yomitoku", help="yomitoku のキャッシュ。cross_ocr_diff と位置推定に使う"),
    ] = None,
    sources: Annotated[
        list[str] | None,
        typer.Option("--source", help=f"使う検出器（複数指定可）。既定は {', '.join(ALL_SOURCES)}"),
    ] = None,
    skip_index: Annotated[
        bool, typer.Option("--skip-index", help="巻末索引が無い本で索引語の検出を省く")
    ] = False,
) -> None:
    """各検出器の誤読候補を集め、原本画像上の位置を推定して候補ファイルに書き出す。"""
    chosen = (
        list(sources) if sources else [s for s in ALL_SOURCES if yomitoku or s != "cross_ocr_diff"]
    )
    try:
        raw_pages = read_raw_pages(pred)
        candidates = collect_candidates(
            raw_pages, yomitoku_dir=yomitoku, sources=chosen, skip_index=skip_index
        )
        if yomitoku is not None:
            candidates = _locate_all(candidates, raw_pages, yomitoku)
        save_candidates(out, candidates)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        raise _fail(str(e)) from e

    table = Table(title=f"誤読候補 {len(candidates)} 件（{len(raw_pages)} ページ）")
    for column in ("検出器", "件数", "位置の特定方法"):
        table.add_column(column)
    for source in ALL_SOURCES:
        subset = [c for c in candidates if c.source == source]
        if subset:
            methods = Counter(c.location for c in subset)
            table.add_row(
                source, str(len(subset)), ", ".join(f"{k}:{v}" for k, v in methods.items())
            )
    Console().print(table)
    Console().print(f"候補を書き出しました: {out}")


def _locate_all(
    candidates: list[JudgeCandidate], raw_pages: dict[int, str], yomitoku: Path
) -> list[JudgeCandidate]:
    """候補の画像上の位置を、yomitoku の段落との対応付けで推定・検証する。"""
    located: list[JudgeCandidate] = []
    cache: dict[int, object] = {}
    for candidate in candidates:
        if candidate.line_index is not None:
            if candidate.page not in cache:
                path = yomitoku / f"page_{candidate.page:04d}" / "analysis.json"
                if not path.is_file():
                    raise FileNotFoundError(f"yomitoku の analysis.json が見つかりません: {path}")
                cache[candidate.page] = parse_analysis(
                    json.loads(path.read_text(encoding="utf-8")), source=str(path)
                )
            candidate = locate_candidate(
                candidate,
                raw_pages[candidate.page],
                cache[candidate.page],  # type: ignore[arg-type]
            )
        located.append(candidate)
    return located


def prepare_images(
    candidates: list[JudgeCandidate],
    renderer: PageRenderer,
    *,
    mode: ImageKind,
    page_max_side: int,
) -> list[tuple[JudgeCandidate, list[bytes], ImageKind]]:
    """候補ごとに LLM へ渡す画像を作る。crop モードで位置が不明な候補はページ全体にする。"""
    items: list[tuple[JudgeCandidate, list[bytes], ImageKind]] = []
    for candidate in candidates:
        page_image = renderer.render(candidate.page)
        if mode == "crop" and candidate.bbox is not None:
            image, kind = crop_region(page_image, candidate.bbox, full_width=True), "crop"
        else:
            image, kind = fit_long_side(page_image, page_max_side), "page"
        items.append((candidate, [encode_jpeg(image)], kind))  # type: ignore[arg-type]
    return items


def summarize_usage(judgments: list[Judgment]) -> dict[str, object]:
    """判定結果のトークン使用量・件数・コストをまとめる。コストは判定ごとのモデルの単価で合計する。"""
    usage = Usage()
    costs: list[float | None] = []
    for j in judgments:
        usage = usage + j.usage
        costs.append(estimate_cost(j.model, j.usage))
    return {
        "judgment_count": len(judgments),
        "error_count": sum(1 for j in judgments if j.error),
        "verdicts": dict(Counter(j.verdict for j in judgments)),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": None
        if any(c is None for c in costs)
        else sum(c for c in costs if c is not None),
    }


@app.command()
def judge(
    candidates_path: Annotated[Path, typer.Option("--candidates", help="collect が書き出した候補")],
    pdf: Annotated[Path, typer.Option("--pdf", help="原本 PDF")],
    out: Annotated[Path, typer.Option("--out", help="判定結果を書き出す JSON ファイル")],
    model: Annotated[
        str, typer.Option("--model", help="判定に使う Gemini のモデル名")
    ] = DEFAULT_MODEL,
    image_mode: Annotated[
        str,
        typer.Option("--image-mode", help="crop（該当箇所の切り出し）または page（ページ全体）"),
    ] = "crop",
    page_max_side: Annotated[
        int, typer.Option("--page-max-side", min=100, help="ページ全体を渡すときの長辺の上限(px)")
    ] = DEFAULT_PAGE_MAX_SIDE,
    crops_dir: Annotated[
        Path | None, typer.Option("--crops-dir", help="LLM に渡した画像を保存するディレクトリ")
    ] = None,
    pages: Annotated[
        list[int] | None, typer.Option("--page", help="このページの候補だけ判定する（複数指定可）")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="先頭 N 件だけ判定する")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = DEFAULT_WORKERS,
) -> None:
    """候補を原本画像と一緒に Gemini に判定させ、判定結果と使用量を書き出す。"""
    if image_mode not in ("crop", "page"):
        raise _fail(f"--image-mode は crop か page です: {image_mode}")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise _fail("環境変数 GEMINI_API_KEY が設定されていません")
    try:
        candidates = load_candidates(candidates_path)
        if pages:
            candidates = [c for c in candidates if c.page in set(pages)]
        if limit is not None:
            candidates = candidates[:limit]
        renderer = PageRenderer(pdf)
        items = prepare_images(candidates, renderer, mode=image_mode, page_max_side=page_max_side)  # type: ignore[arg-type]
        if crops_dir is not None:
            crops_dir.mkdir(parents=True, exist_ok=True)
            for candidate, images, _ in items:
                (crops_dir / f"{candidate.id}.jpg").write_bytes(images[0])
        client = GeminiLlmClient(model=model, api_key=api_key)
        judgments = judge_all(client, items, workers=workers)
    except (FileNotFoundError, ValueError, OSError) as e:
        raise _fail(str(e)) from e

    summary = summarize_usage(judgments)
    result = {
        "format_version": JUDGMENTS_FORMAT_VERSION,
        "model": model,
        "image_mode": image_mode,
        "summary": summary,
        "judgments": [j.to_dict() for j in judgments],
        "image_kinds": {c.id: kind for c, _, kind in items},
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    console = Console()
    console.print(f"判定 {summary['judgment_count']} 件 / 判定結果: {summary['verdicts']}")
    console.print(
        f"入力 {summary['input_tokens']:,} トークン・出力 {summary['output_tokens']:,} トークン・"
        f"推定コスト {_fmt(summary['cost_usd'])} USD"  # type: ignore[arg-type]
    )
    console.print(f"判定結果を書き出しました: {out}")
    if summary["error_count"]:
        raise _fail(f"{summary['error_count']} 件の判定に失敗しました（{out} の error を参照）")


def _load_judgments(path: Path) -> tuple[str, list[Judgment]]:
    """judge コマンドが書き出した判定結果を読み込む（モデル名と判定のリスト）。"""
    if not path.is_file():
        raise FileNotFoundError(f"判定結果が見つかりません: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != JUDGMENTS_FORMAT_VERSION:
        raise ValueError(
            f"判定結果の format_version は {JUDGMENTS_FORMAT_VERSION} である必要があります: {path}"
        )
    return data["model"], [Judgment.from_dict(j) for j in data["judgments"]]


@dataclass(frozen=True)
class RunSet:
    """候補に共通して判定済みの、1 つ以上の判定結果（run）の集まり。

    Attributes:
        key: 表示用の名前（ファイル名を + でつないだもの）。
        candidates: 全ての run で判定済みの候補。
        runs: run ごとの判定（candidates に絞ったもの）。
        flat: 全 run の判定をつないだもの（コストの集計用）。
    """

    key: str
    candidates: list[JudgeCandidate]
    runs: list[list[Judgment]]
    flat: list[Judgment]


def _load_run_set(spec: str, candidates: list[JudgeCandidate]) -> RunSet:
    """`a.json` または `a.json+b.json`（複数 run の合意をとる）を読み込む。"""
    parts = spec.split("+")
    loaded = [_load_judgments(Path(part))[1] for part in parts]
    common = set.intersection(*({j.candidate_id for j in judgments} for judgments in loaded))
    subset = [c for c in candidates if c.id in common]
    runs = [[j for j in judgments if j.candidate_id in common] for judgments in loaded]
    key = "+".join(Path(part).stem for part in parts)
    return RunSet(key=key, candidates=subset, runs=runs, flat=[j for run in runs for j in run])


def _policy(
    min_confidence: str,
    max_edit_chars: int,
    auto_sources: list[str] | None,
    min_agree: int | None,
    run_count: int,
) -> AutoPolicy:
    """自動適用の基準を作る。min_agree が未指定なら、全ての run の一致を要求する。"""
    return AutoPolicy(
        min_confidence=min_confidence,  # type: ignore[arg-type]
        max_edit_chars=max_edit_chars,
        auto_sources=frozenset(auto_sources) if auto_sources else AutoPolicy().auto_sources,
        min_agree=min_agree if min_agree is not None else run_count,
    )


@app.command()
def export(
    candidates_path: Annotated[Path, typer.Option("--candidates")],
    judgments_paths: Annotated[
        list[Path],
        typer.Option(
            "--judgments", help="判定結果（複数指定すると、一致したものだけを自動適用する）"
        ),
    ],
    pred: Annotated[
        Path, typer.Option("--pred", help="候補を集めたときと同じ Gemini の出力/キャッシュ")
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="修正ファイルの出力先")],
    min_confidence: Annotated[
        str, typer.Option("--min-confidence", help="自動適用に必要な確信度")
    ] = "high",
    max_edit_chars: Annotated[
        int, typer.Option("--max-edit-chars", min=1, help="自動適用できる置換の最大変更文字数")
    ] = DEFAULT_MAX_EDIT_CHARS,
    auto_sources: Annotated[
        list[str] | None, typer.Option("--auto-source", help="自動適用を許す検出器（複数指定可）")
    ] = None,
    min_agree: Annotated[
        int | None,
        typer.Option("--min-agree", min=1, help="自動適用に必要な一致数（既定: 判定結果の数）"),
    ] = None,
) -> None:
    """判定結果を、自動適用する修正と人の確認待ちの修正（#28 の形式）に振り分けて書き出す。"""
    try:
        run_set = _load_run_set(
            "+".join(str(p) for p in judgments_paths), load_candidates(candidates_path)
        )
        exports = build_exports(
            run_set.candidates,
            run_set.runs,
            read_raw_pages(pred),
            _policy(min_confidence, max_edit_chars, auto_sources, min_agree, len(run_set.runs)),
            date=datetime.date.today().isoformat(),
        )
        write_exports(exports, out_dir)
    except (FileNotFoundError, ValueError, OSError, KeyError) as e:
        raise _fail(str(e)) from e

    counts = Counter(d["decision"] for d in exports.decisions)
    Console().print(
        f"候補 {len(exports.decisions)} 件: 自動適用 {counts['auto']}・確認待ち {counts['review']}・"
        f"棄却 {counts['reject']}（確認待ちのうち修正案あり {len(exports.pending)}）"
    )
    Console().print(f"書き出しました: {out_dir}")


@app.command()
def evaluate(
    truth: Annotated[Path, typer.Option("--truth", help="評価セット")],
    pred: Annotated[
        Path, typer.Option("--pred", help="人手修正前の Gemini キャッシュ（ベースライン）")
    ],
    candidates_path: Annotated[Path, typer.Option("--candidates")],
    judgments_specs: Annotated[
        list[str],
        typer.Option(
            "--judgments",
            help="判定結果。`a.json+b.json` のように + でつなぐと、合意（全 run の一致）で自動適用する。複数指定可",
        ),
    ],
    known_corrections: Annotated[
        Path,
        typer.Option(
            "--known-corrections",
            help="既知の誤読の人手修正ファイル（正解ラベルと上限の比較に使う）",
        ),
    ],
    labels_path: Annotated[
        Path | None,
        typer.Option("--labels", help="手動の正解ラベル JSON（{候補 id: misread|correct}）"),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="結果を書き出す JSON")] = None,
    min_confidence: Annotated[str, typer.Option("--min-confidence")] = "high",
    max_edit_chars: Annotated[
        int, typer.Option("--max-edit-chars", min=1)
    ] = DEFAULT_MAX_EDIT_CHARS,
    auto_sources: Annotated[list[str] | None, typer.Option("--auto-source")] = None,
    min_agree: Annotated[int | None, typer.Option("--min-agree", min=1)] = None,
) -> None:
    """評価セットで、判定の正誤・修正適用後の指標・コストを判定結果ごとに測る。"""
    try:
        manifest_pages = {entry.page for entry in load_manifest(truth)}
        raw_all = read_raw_pages(pred)
        raw_pages = {p: t for p, t in raw_all.items() if p in manifest_pages}
        known = [c for items in load_corrections(known_corrections).values() for c in items]
        manual = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path else {}
        all_candidates = load_candidates(candidates_path)
        candidates = [c for c in all_candidates if c.page in manifest_pages]
        # 評価セット外でも、手動ラベル（原本画像で人が確認した結果）のある候補は別枠で集計する
        extra_candidates = [
            c for c in all_candidates if c.page not in manifest_pages and c.id in manual
        ]
        labels = derive_labels(candidates, known, manual, raw_pages)
        labels_extra = {c.id: manual[c.id] for c in extra_candidates}
        results: dict[str, Any] = {
            "page_count": len(manifest_pages),
            "candidate_count": len(candidates),
            "extra_candidate_count": len(extra_candidates),
            "labels": labels,
            "labels_extra": labels_extra,
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            scores = {
                "baseline": measure_scores(
                    truth, raw_pages, [], work / "baseline", katex_checker=check_katex
                ),
                "known_corrections": measure_scores(
                    truth, raw_pages, known, work / "known", katex_checker=check_katex
                ),
            }
            models: dict[str, Any] = {}
            for spec in judgments_specs:
                run_set = _load_run_set(spec, [*candidates, *extra_candidates])
                policy = _policy(
                    min_confidence, max_edit_chars, auto_sources, min_agree, len(run_set.runs)
                )
                key = run_set.key
                usage = summarize_usage(
                    [j for j in run_set.flat if j.candidate_id in {c.id for c in candidates}]
                )
                parts: dict[str, Any] = {}
                for label, group, group_labels, group_raw, group_known in (
                    ("eval", candidates, labels, raw_pages, known),
                    ("extra", extra_candidates, labels_extra, raw_all, []),
                ):
                    ids = {c.id for c in group} & {c.id for c in run_set.candidates}
                    group_candidates = [c for c in run_set.candidates if c.id in ids]
                    group_runs = [[j for j in run if j.candidate_id in ids] for run in run_set.runs]
                    by_run = [{j.candidate_id: j for j in run} for run in group_runs]
                    combined = [
                        combine([by_id[c.id] for by_id in by_run], c, group_raw[c.page]).judgment
                        for c in group_candidates
                    ]
                    accuracy = accuracy_report(
                        group_candidates, combined, group_labels, group_raw, group_known
                    )
                    exports = build_exports(
                        group_candidates, group_runs, group_raw, policy, date="evaluation"
                    )
                    auto_ids = {
                        d["candidate_id"] for d in exports.decisions if d["decision"] == "auto"
                    }
                    parts[label] = (accuracy, exports)
                    models.setdefault(key, {})[f"accuracy_{label}"] = accuracy.to_dict()
                    models[key][f"rows_{label}"] = accuracy.rows
                    models[key][f"decisions_{label}"] = dict(
                        Counter(d["decision"] for d in exports.decisions)
                    )
                    # 自動適用と判断されたが、正解ラベルが「正しい（誤検出）」の候補。誤った修正が入る危険な結果
                    models[key][f"wrong_auto_{label}"] = [
                        c.id
                        for c in group_candidates
                        if c.id in auto_ids and group_labels[c.id] == "correct"
                    ]
                exports = parts["eval"][1]
                scores[f"{key}:auto"] = measure_scores(
                    truth, raw_pages, exports.auto, work / f"{key}-auto", katex_checker=check_katex
                )
                scores[f"{key}:auto+pending"] = measure_scores(
                    truth,
                    raw_pages,
                    [*exports.auto, *exports.pending],
                    work / f"{key}-all",
                    katex_checker=check_katex,
                )
                models[key].update(
                    {
                        "usage": usage,
                        "auto_corrections": len(exports.auto),
                        "pending_corrections": len(exports.pending),
                        "cost_per_page_usd": _per(usage["cost_usd"], len(manifest_pages)),
                        "cost_per_candidate_usd": _per(usage["cost_usd"], len(candidates)),
                    }
                )
            results["scores"] = scores
            results["models"] = models
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        KeyError,
        KatexCheckError,
        CorrectionError,
    ) as e:
        raise _fail(str(e)) from e

    _print_evaluation(results)
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        Console().print(f"結果を書き出しました: {out}")


def _per(total: object, count: int) -> float | None:
    return total / count if isinstance(total, float) and count else None


def _print_evaluation(results: dict[str, Any]) -> None:
    console = Console()
    for label, title in (
        (
            "eval",
            f"評価セット {results['page_count']} ページ・候補 {results['candidate_count']} 件",
        ),
        ("extra", f"評価セット外の確認済み候補 {results['extra_candidate_count']} 件"),
    ):
        table = Table(title=f"判定の正誤（{title}）")
        for column in (
            "判定",
            "真の誤読",
            "TP",
            "修正も正しい",
            "FN",
            "誤検出",
            "FP",
            "TN",
            "uncertain",
            "自動適用",
            "うち誤り",
        ):
            table.add_column(column)
        for key, m in results["models"].items():
            a = m[f"accuracy_{label}"]
            table.add_row(
                key,
                str(a["misread_count"]),
                str(a["true_positive"]),
                str(a["fix_correct"]),
                str(a["false_negative"]),
                str(a["correct_count"]),
                str(a["false_positive"]),
                str(a["true_negative"]),
                str(a["uncertain_on_correct"]),
                str(m[f"decisions_{label}"].get("auto", 0)),
                str(len(m[f"wrong_auto_{label}"])),
            )
        console.print(table)
    table = Table(title="コスト（評価セットの候補）")
    for column in ("判定", "コスト/候補(USD)", "コスト/ページ(USD)"):
        table.add_column(column)
    for key, m in results["models"].items():
        table.add_row(key, _fmt(m["cost_per_candidate_usd"]), _fmt(m["cost_per_page_usd"]))
    console.print(table)
    table = Table(title="修正適用後の指標（評価セット全体）")
    for column in ("条件", "地の文CER", "数式F1", "数式CER", "見出しF1", "KaTeXエラー"):
        table.add_column(column)
    for name, s in results["scores"].items():
        table.add_row(
            name,
            f"{s['prose_cer']:.2%}",
            f"{s['math_f1']:.3f}",
            f"{s['math_cer']:.2%}",
            f"{s['heading_f1']:.3f}",
            str(int(s["katex_errors"])),
        )
    console.print(table)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.5f}"


if __name__ == "__main__":  # pragma: no cover
    app()
