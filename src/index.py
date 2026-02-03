from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndexMeta:
    model_name: str
    n_items: int
    columns: list[str]


def save_index(embeddings: np.ndarray, df: pd.DataFrame, meta: IndexMeta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    df.to_csv(out_dir / "items.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": meta.model_name, "n_items": meta.n_items, "columns": meta.columns},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_index(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    emb_path = index_dir / "embeddings.npy"
    items_path = index_dir / "items.csv"
    meta_path = index_dir / "meta.json"
    if not emb_path.exists() or not items_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index artifacts not found in {index_dir}")
    embeddings = np.load(emb_path)
    items = pd.read_csv(items_path, dtype=str).fillna("")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, items, meta
