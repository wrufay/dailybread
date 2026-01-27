"""
FastAPI server for semantic Bible verse search.
Uses numpy for similarity search (no FAISS dependency issues).
"""

import os
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Global state
model: SentenceTransformer = None
embeddings: np.ndarray = None
metadata: list[dict] = None

BASE_PATH = Path(__file__).parent
MODEL_NAME = "all-MiniLM-L6-v2"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class VerseResult(BaseModel):
    reference: str
    book: str
    chapter: int
    verse: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[VerseResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and embeddings on startup."""
    global model, embeddings, metadata

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading embeddings...")
    embeddings_path = BASE_PATH / "embeddings.npy"
    if not embeddings_path.exists():
        raise RuntimeError("embeddings.npy not found. Run: python3 embed.py")
    embeddings = np.load(embeddings_path)

    print("Loading metadata...")
    metadata_path = BASE_PATH / "verses_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("verses_metadata.json not found. Run: python3 embed.py")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Ready! {len(metadata)} verses indexed.")
    yield

    print("Shutting down...")


app = FastAPI(
    title="FirstLoved Bible Search",
    description="Semantic search for Bible verses",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "verses_indexed": len(metadata) if metadata else 0}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for verses by meaning."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.top_k < 1 or request.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    # Embed the query
    query_embedding = model.encode(
        request.query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Cosine similarity (dot product since normalized)
    scores = np.dot(embeddings, query_embedding)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    # Build results
    results = []
    for idx in top_indices:
        verse = metadata[idx]
        results.append(VerseResult(
            reference=verse["reference"],
            book=verse["book"],
            chapter=verse["chapter"],
            verse=verse["verse"],
            text=verse["text"],
            score=float(scores[idx])
        ))

    return SearchResponse(query=request.query, results=results)


@app.get("/search")
async def search_get(query: str, top_k: int = 10):
    """GET endpoint for easier testing."""
    return await search(SearchRequest(query=query, top_k=top_k))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
