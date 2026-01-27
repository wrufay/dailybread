# FirstLoved Bible Semantic Search Backend

Semantic search API for Bible verses using sentence-transformers and numpy.

## Setup

```bash
pip install -r requirements.txt
```

## Build the Index

1. **Fetch all Bible verses**:
   ```bash
   python3 fetch_verses.py
   ```
   Downloads KJV Bible and creates `verses.json` with 31K+ verses.

2. **Build embeddings**:
   ```bash
   python3 embed.py
   ```
   Creates `embeddings.npy` and `verses_metadata.json`.

## Run the Server

```bash
python3 -m uvicorn main:app --reload
```

Server runs at `http://localhost:8000`

## API Endpoints

### Search
```bash
# POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "I feel anxious", "top_k": 5}'

# GET
curl "http://localhost:8000/search?query=finding+peace&top_k=5"
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Deploy to Railway

1. Push to GitHub (use Git LFS for `embeddings.npy`)
2. Connect repo to Railway
3. Railway auto-detects Dockerfile and deploys

## Files

- `fetch_verses.py` - Downloads KJV Bible from GitHub
- `embed.py` - Creates embeddings using sentence-transformers
- `main.py` - FastAPI server with /search endpoint
- `verses.json` - Raw Bible verse data
- `embeddings.npy` - Verse embeddings (numpy array)
- `verses_metadata.json` - Verse text and references for lookup
