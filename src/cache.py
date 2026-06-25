import hashlib
import json
import time
from pathlib import Path
from typing import Optional


def cache_key(text: str, model: str, strategy: str, temperature: float = 0.0) -> str:
    raw = f"{model}||{strategy}||{temperature}||{text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_load(cache_dir: Path, key: str) -> Optional[dict]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def cache_save(cache_dir: Path, key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        **data,
        "timestamp": time.time(),
    }
    path = cache_dir / f"{key}.json"
    with open(path, "w") as f:
        json.dump(entry, f, ensure_ascii=False)
