"""仕様: 判定モデルの単価表と、トークン使用量からのコスト見積もり。

単価は Gemini API の標準料金（有料枠、バッチ・キャッシュなし）を 2026-10-10 に
https://ai.google.dev/gemini-api/docs/pricing で確認したもの。USD / 100 万トークン。
注意:
  - gemini-3.8-flash / gemini-3.5-flash の料金は 2026-12-31 までの特別価格で、2027-01-01 から 2 倍になる
  - gemini-3.1-pro-preview は入力 20 万トークン以下の単価（本タスクの 1 回の呼び出しは数千トークン）
  - 出力単価は思考トークンを含む
単価が表に無いモデルは見積もれないので None を返す（暗黙の既定値は使わない）。
"""

from __future__ import annotations

from ouj_notebook_converter.judge.models import Usage

TOKENS_PER_UNIT = 1_000_000

# モデル名 → (入力単価, 出力単価) [USD / 100 万トークン]
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    """トークン使用量から API 料金（USD）を見積もる。単価が不明なモデルは None。"""
    prices = MODEL_PRICES.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / TOKENS_PER_UNIT
