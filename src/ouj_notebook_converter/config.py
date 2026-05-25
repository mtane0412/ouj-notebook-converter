"""仕様: pydantic-settings を使ったアプリケーション設定クラス。

.env ファイルおよび環境変数から設定を読み込む。
優先順位: CLI オプション > 環境変数 > .env ファイル。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定。

    .env ファイルおよび環境変数から自動的に読み込む。
    環境変数が .env ファイルより優先される。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = None
