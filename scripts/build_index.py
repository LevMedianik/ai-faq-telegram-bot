from __future__ import annotations

from src.config import settings
from src.dataio import load_variants
from src.embedder import Embedder
from src.index import IndexMeta, save_index


def main():
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variants(settings.variants_csv)
    texts = variants["question_variant"].tolist()

    embedder = Embedder(settings.model_name)
    emb = embedder.encode(texts)

    meta = IndexMeta(
        model_name=settings.model_name,
        n_items=int(emb.shape[0]),
        columns=list(variants.columns),
    )
    save_index(emb, variants, meta, settings.artifacts_dir)

    print(f"   Index built: {settings.artifacts_dir}")
    print(f"   items: {emb.shape[0]}, dim: {emb.shape[1]}")
    print(f"   model: {settings.model_name}")


if __name__ == "__main__":
    main()
