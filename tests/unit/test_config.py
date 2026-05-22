"""仕様: Settings クラス（pydantic-settings ベースの設定読み込み）のユニットテスト。

.env ファイルおよび環境変数からの設定読み込みを検証する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ouj_notebook_converter.config import Settings


class TestSettings:
    """Settings クラスの動作検証。"""

    def test_envファイルからGEMINI_API_KEYを読み込む(self, tmp_path: Path) -> None:
        """.env ファイルに書かれた GEMINI_API_KEY が読み込まれること。"""
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=テスト用APIキー\n", encoding="utf-8")

        settings = Settings(_env_file=env_file)

        assert settings.gemini_api_key == "テスト用APIキー"

    def test_GEMINI_API_KEY未設定時はNone(self, tmp_path: Path) -> None:
        """GEMINI_API_KEY が設定されていない場合に None を返すこと。"""
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")

        settings = Settings(_env_file=env_file)

        assert settings.gemini_api_key is None

    def test_envファイルが存在しない場合はNone(self, tmp_path: Path) -> None:
        """.env ファイルが存在しなくても正常に初期化され None を返すこと。"""
        settings = Settings(_env_file=tmp_path / "存在しない.env")

        assert settings.gemini_api_key is None

    def test_環境変数がenvファイルより優先される(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """環境変数 GEMINI_API_KEY が .env ファイルより優先されること。"""
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=envファイルのキー\n", encoding="utf-8")
        monkeypatch.setenv("GEMINI_API_KEY", "環境変数のキー")

        settings = Settings(_env_file=env_file)

        assert settings.gemini_api_key == "環境変数のキー"
