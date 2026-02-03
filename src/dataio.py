from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FAQItem:
    faq_id: str
    canonical_question: str
    answer: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    # Use pandas robust CSV parsing; quotes handled automatically
    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    return df


def load_faq(path: Path) -> list[FAQItem]:
    df = _read_csv(path)
    required = {"faq_id", "canonical_question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faq.csv missing columns: {missing}. Found: {list(df.columns)}")
    items: list[FAQItem] = []
    for _, row in df.iterrows():
        faq_id = row["faq_id"].strip()
        if not faq_id:
            continue
        items.append(
            FAQItem(
                faq_id=faq_id,
                canonical_question=row["canonical_question"].strip(),
                answer=row["answer"].strip(),
            )
        )
    return items


def load_variants(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    required = {"faq_id", "question_variant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"variants.csv missing columns: {missing}. Found: {list(df.columns)}"
        )
    df["faq_id"] = df["faq_id"].astype(str).str.strip()
    df["question_variant"] = df["question_variant"].astype(str).str.strip()
    df = df[(df["faq_id"] != "") & (df["question_variant"] != "")]
    return df.reset_index(drop=True)
