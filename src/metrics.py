from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class EvalResult:
    accuracy_at_1: float
    precision: float
    recall: float
    coverage: float  # fraction of queries answered confidently


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str | None],
) -> EvalResult:
    y_true = list(y_true)
    y_pred_list = list(y_pred)

    answered_mask = [p is not None for p in y_pred_list]
    coverage = float(np.mean(answered_mask)) if y_pred_list else 0.0

    # For metric computation, treat None as special label "__NONE__"
    y_pred_filled = [p if p is not None else "__NONE__" for p in y_pred_list]

    acc = accuracy_score(y_true, y_pred_filled)

    # precision/recall over "answered correctly" in multiclass setting:
    # use micro averages, but include "__NONE__" as class to penalize abstention.
    prec = precision_score(y_true, y_pred_filled, average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred_filled, average="micro", zero_division=0)

    return EvalResult(
        accuracy_at_1=float(acc),
        precision=float(prec),
        recall=float(rec),
        coverage=coverage,
    )


def make_split_per_faq(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []

    for faq_id, group in df.groupby("faq_id"):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        # Ensure at least 1 train if possible
        if len(train_idx) == 0 and len(idx) > 1:
            train_idx = idx[:1]
            test_idx = idx[1:]
        train_rows.append(df.loc[train_idx])
        test_rows.append(df.loc[test_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df
