"""
Build search index from Bible verses using sentence-transformers.
Uses numpy for similarity search (simpler, works everywhere).
"""

import os
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_verses(path: str = "verses.json") -> list[dict]:
    """Load verses from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    import sys
    print("Starting...", flush=True)

    base_path = Path(__file__).parent

    # Check if verses.json exists
    verses_path = base_path / "verses.json"
    if not verses_path.exists():
        print("verses.json not found!")
        print("Run: python3 fetch_verses.py")
        return

    # Load verses
    print("Loading verses...", flush=True)
    verses = load_verses(verses_path)
    print(f"Loaded {len(verses)} verses", flush=True)

    # Load model
    print(f"Loading model: {MODEL_NAME}", flush=True)
    print("(This downloads ~90MB on first run)", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded!", flush=True)

    # Generate embeddings
    print(f"Embedding {len(verses)} verses...")
    texts = [v["text"] for v in verses]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save embeddings as numpy array
    embeddings_path = base_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

    # Save verse metadata
    metadata_path = base_path / "verses_metadata.json"
    metadata = [
        {
            "reference": v["reference"],
            "book": v["book"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "text": v["text"]
        }
        for v in verses
    ]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {metadata_path}")

    print("\nDone! You can now run: python3 -m uvicorn main:app --reload")

if __name__ == "__main__":
    main()
