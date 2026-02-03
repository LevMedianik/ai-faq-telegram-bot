from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _project_root() -> Path:
    # src/config.py -> src -> project root
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = _project_root()

    # Data
    data_dir: Path = root / "data"
    faq_csv: Path = data_dir / "faq.csv"
    variants_csv: Path = data_dir / "variants.csv"
    splits_json: Path = data_dir / "splits.json"

    # Artifacts
    artifacts_dir: Path = root / "artifacts"
    embeddings_path: Path = artifacts_dir / "embeddings.npy"
    meta_path: Path = artifacts_dir / "meta.json"

    # Model / retrieval
    model_name: str = os.getenv(
        "MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    threshold: float = float(os.getenv("THRESHOLD", "0.55"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Bot
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")


settings = Settings()
