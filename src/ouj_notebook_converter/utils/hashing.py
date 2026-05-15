"""仕様: ファイルおよび文字列の SHA-256 ハッシュを算出するユーティリティ。"""
import hashlib
from pathlib import Path


def sha256_string(text: str) -> str:
    """文字列の SHA-256 ハッシュを 64 文字の 16 進数で返す。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, short: bool = False) -> str:
    """ファイルの SHA-256 ハッシュを返す。

    Args:
        path: ハッシュ対象のファイルパス。
        short: True の場合、先頭 8 文字のみ返す（ディレクトリ名等に使用）。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:8] if short else digest
