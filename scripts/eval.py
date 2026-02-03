from __future__ import annotations

import json

from src.config import settings
from src.dataio import load_faq, load_variants
from src.embedder import Embedder
from src.index import save_index, IndexMeta
from src.metrics import evaluate_predictions, make_split_per_faq
from src.retriever import Retriever


def main():
    faq = load_faq(settings.faq_csv)
    faq_map = {x.faq_id: x for x in faq}

    variants = load_variants(settings.variants_csv)

    # Load split config
    with open(settings.splits_json, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)
    test_ratio = float(split_cfg.get("test_ratio", 0.2))
    seed = int(split_cfg.get("random_seed", 42))

    train_df, test_df = make_split_per_faq(variants, test_ratio=test_ratio, seed=seed)

    # Build embeddings for TRAIN variants only
    embedder = Embedder(settings.model_name)
    train_texts = train_df["question_variant"].tolist()
    train_emb = embedder.encode(train_texts)

    # Save artifacts for reproducibility (optional but useful)
    meta = IndexMeta(settings.model_name, int(train_emb.shape[0]), list(train_df.columns))
    save_index(train_emb, train_df, meta, settings.artifacts_dir)

    retriever = Retriever(train_emb, train_df, threshold=settings.threshold, top_k=settings.top_k)

    # Evaluate on TEST: embed each test query, predict faq_id or None
    y_true = test_df["faq_id"].tolist()
    y_pred = []
    scores = []

    test_texts = test_df["question_variant"].tolist()
    test_emb = embedder.encode(test_texts)

    for i in range(test_emb.shape[0]):
        pred_id, score, confident = retriever.predict_faq(test_emb[i])
        y_pred.append(pred_id)
        scores.append(score)

    res = evaluate_predictions(y_true, y_pred)

    print("    Eval results")
    print(f"   model      : {settings.model_name}")
    print(f"   threshold  : {settings.threshold}")
    print(f"   top_k      : {settings.top_k}")
    print(f"   test_ratio : {test_ratio}")
    print(f"   accuracy@1 : {res.accuracy_at_1:.4f}")
    print(f"   precision  : {res.precision:.4f}")
    print(f"   recall     : {res.recall:.4f}")
    print(f"   coverage   : {res.coverage:.4f}  (answered confidently)")

    # Quick sanity: show a few low-score cases
    worst = sorted(range(len(scores)), key=lambda j: scores[j])[:5]
    print("\nLowest-confidence examples:")
    for j in worst:
        print(f"- true={y_true[j]} score={scores[j]:.3f} text='{test_texts[j]}'")
        if y_pred[j] is None:
            print("  pred=None (sent to operator)")
        else:
            print(f"  pred={y_pred[j]} answer='{faq_map.get(y_pred[j]).answer if y_pred[j] in faq_map else ''}'")

# Текущие параметры:
# - threshold: 0.55
# - top_k: 5

if __name__ == "__main__":
    main()
