from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RetrievalHit:
    row_idx: int
    faq_id: str
    score: float
    question_variant: str


class Retriever:
    def __init__(self, embeddings: np.ndarray, items_df, threshold: float, top_k: int = 5):
        """
        embeddings: (N, D) float32, normalized
        items_df: dataframe with columns ['faq_id', 'question_variant']
        """
        self.emb = embeddings
        self.items = items_df
        self.threshold = float(threshold)
        self.top_k = int(top_k)

        if len(self.items) != self.emb.shape[0]:
            raise ValueError("Embeddings count != items rows")

    @staticmethod
    def _cosine_scores(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
        # embeddings are normalized => cosine = dot product
        return matrix @ vec

    def query(self, query_emb: np.ndarray) -> list[RetrievalHit]:
        scores = self._cosine_scores(self.emb, query_emb)
        k = min(self.top_k, scores.shape[0])
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        hits: list[RetrievalHit] = []
        for i in idxs:
            hits.append(
                RetrievalHit(
                    row_idx=int(i),
                    faq_id=str(self.items.iloc[i]["faq_id"]),
                    score=float(scores[i]),
                    question_variant=str(self.items.iloc[i]["question_variant"]),
                )
            )
        return hits

    def predict_faq(self, query_emb: np.ndarray) -> tuple[str | None, float, bool]:
        hits = self.query(query_emb)
        if not hits:
            return None, 0.0, False
        best = hits[0]
        is_confident = best.score >= self.threshold
        return (best.faq_id if is_confident else None), best.score, is_confident
