"""仕様: 検出器が挙げた誤読候補を原本画像と照合して LLM（Gemini）に判定させるパッケージ（issue #24）。

`python -m ouj_notebook_converter.judge {collect,judge,export,evaluate}` で実行する。
流れは README の「誤読候補の LLM 判定」を参照。
"""
