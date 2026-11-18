"""FastAPI AI Assistant for DarGlobal and Wasalt Real Estate.
Features:
- Merges data/docs.jsonl and data/seed.jsonl at startup and on reload
- Dual-script BM25 tokenization with site-aware query boosting and balanced top-6 retrieval
- Graceful response when no context matches (zero model calls with empty context)
- OpenRouter multi-model fallback on 404, 429, and 5xx errors
- Strict header formatting and quote/whitespace stripping for OPENROUTER_API_KEY
- follow_redirects disabled to prevent stripping of Authorization header
- /health reporting chunks per site, models, key status, key length, and key prefix
- /reload protected by ADMIN_TOKEN
- Background scraper launch if docs.jsonl has fewer than 100 chunks
"""
import asyncio, json, os, re, subprocess, sys, threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

DOCS_FILE = Path(os.getenv("DATA_FILE", "data/docs.jsonl"))
SEED_FILE = Path(os.getenv("SEED_FILE", "data/seed.jsonl"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

DEFAULT_MODELS = (
    "meta-llama/llama-3.3-70b-instruct:free,"
    "google/gemini-2.0-flash-exp:free,"
    "qwen/qwen-2.5-72b-instruct:free,"
    "mistralai/mistral-small-24b-instruct-2501:free,"
    "google/gemma-3-27b-it:free"
)
MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", DEFAULT_MODELS).split(",") if m.strip()]

SYSTEM_PROMPT = (
    "You are an expert real estate AI assistant for DarGlobal and Wasalt.\n"
    "Answer questions factually and concisely using ONLY the provided context.\n"
    "Explicitly mention whether you are referring to DarGlobal or Wasalt.\n"
    "If the context does not contain enough information, state that you do not have that specific information."
)

DOCS: list[dict] = []
BM: Optional[BM25Okapi] = None
CHUNKS_PER_SITE: dict[str, int] = {"DarGlobal": 0, "Wasalt": 0}
LOCK = threading.Lock()

def get_openrouter_api_key() -> str:
    """Read OPENROUTER_API_KEY dynamically, stripping whitespace, newlines, and quotes."""
    raw = os.getenv("OPENROUTER_API_KEY", "")
    if not raw:
        return ""
    cleaned = raw.strip()
    while (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned

def tok(s: str) -> list[str]:
    # Match alphanumeric words in Latin and Arabic scripts
    return re.findall(r"[\w\u0600-\u06FF]+", s.lower())

def load_and_merge_docs() -> tuple[list[dict], dict[str, int]]:
    combined = []
    seen = set()

    for file_path in [DOCS_FILE, SEED_FILE]:
        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            # Deduplicate by URL and beginning of text
                            key = (d.get("url", ""), d.get("text", "")[:100].strip())
                            if key not in seen:
                                seen.add(key)
                                combined.append(d)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    counts = {"DarGlobal": 0, "Wasalt": 0}
    for d in combined:
        site = d.get("site", "Other")
        counts[site] = counts.get(site, 0) + 1

    return combined, counts

def reload_data_index():
    global DOCS, BM, CHUNKS_PER_SITE
    with LOCK:
        DOCS, CHUNKS_PER_SITE = load_and_merge_docs()
        if DOCS:
            tokenized_corpus = [tok(d.get("title", "") + " " + d.get("text", "")) for d in DOCS]
            BM = BM25Okapi(tokenized_corpus)
        else:
            BM = None
        print(f"[Index] Loaded {len(DOCS)} merged chunks: {CHUNKS_PER_SITE}")

# Initial load
reload_data_index()

def run_background_scrape_if_needed():
    docs_count = 0
    if DOCS_FILE.exists():
        try:
            with DOCS_FILE.open("r", encoding="utf-8") as f:
                docs_count = sum(1 for line in f if line.strip())
        except Exception:
            docs_count = 0

    if docs_count < 100:
        print(f"[Startup] docs.jsonl has {docs_count} chunks (< 100). Starting background scraper...")
        def _scrape_job():
            try:
                proc = subprocess.run([sys.executable, "-m", "scraper.scrape"], check=False)
                print(f"[Startup Background Scraper] Exited with code {proc.returncode}")
                reload_data_index()
            except Exception as e:
                print(f"[Startup Background Scraper] Error: {e}")

        t = threading.Thread(target=_scrape_job, daemon=True)
        t.start()
    else:
        print(f"[Startup] docs.jsonl already has {docs_count} chunks (>= 100). No startup scrape needed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    run_background_scrape_if_needed()
    yield

app = FastAPI(title="DarGlobal and Wasalt Real Estate Assistant", lifespan=lifespan)

class Msg(BaseModel):
    role: str
    content: str

class Req(BaseModel):
    message: str
    history: list[Msg] = []

def detect_site_filter(q: str) -> str:
    q_lower = q.lower()
    has_dar = bool(re.search(r"\b(darglobal|dar\s*global)\b", q_lower))
    has_wasalt = bool(re.search(r"\b(wasalt|waslat|وصلت)\b", q_lower))
    if has_dar and not has_wasalt:
        return "DarGlobal"
    if has_wasalt and not has_dar:
        return "Wasalt"
    return "both"

def retrieve(query: str, k: int = 6) -> list[dict]:
    with LOCK:
        if not BM or not DOCS:
            return []
        tokens = tok(query)
        if not tokens:
            return []
        scores = BM.get_scores(tokens)

    site_filter = detect_site_filter(query)

    if site_filter == "DarGlobal":
        boosted = []
        for i, doc in enumerate(DOCS):
            s = scores[i]
            if s > 0:
                boost = 1.6 if doc.get("site") == "DarGlobal" else 0.8
                boosted.append((doc, s * boost))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in boosted[:k]]

    elif site_filter == "Wasalt":
        boosted = []
        for i, doc in enumerate(DOCS):
            s = scores[i]
            if s > 0:
                boost = 1.6 if doc.get("site") == "Wasalt" else 0.8
                boosted.append((doc, s * boost))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in boosted[:k]]

    else:
        dar_hits = []
        wasalt_hits = []
        other_hits = []

        for i, doc in enumerate(DOCS):
            s = scores[i]
            if s > 0:
                site = doc.get("site")
                if site == "DarGlobal":
                    dar_hits.append((doc, s))
                elif site == "Wasalt":
                    wasalt_hits.append((doc, s))
                else:
                    other_hits.append((doc, s))

        dar_hits.sort(key=lambda x: x[1], reverse=True)
        wasalt_hits.sort(key=lambda x: x[1], reverse=True)

        selected = [doc for doc, _ in dar_hits[:3]] + [doc for doc, _ in wasalt_hits[:3]]

        if len(selected) < k:
            remaining = [doc for doc, _ in dar_hits[3:]] + [doc for doc, _ in wasalt_hits[3:]] + [doc for doc, _ in other_hits]
            remaining.sort(key=lambda doc: scores[DOCS.index(doc)], reverse=True)
            for doc in remaining:
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= k:
                    break

        return selected

@app.get("/health")
def health():
    key = get_openrouter_api_key()
    return {
        "ok": True,
        "chunks_per_site": CHUNKS_PER_SITE,
        "total_chunks": len(DOCS),
        "models": MODELS,
        "has_api_key": bool(key),
        "key_length": len(key),
        "key_prefix": key[:8] if key else "",
    }

@app.post("/reload")
def reload_index(
    authorization: Optional[str] = Header(None),
    x_admin_token: Optional[str] = Header(None),
    token: Optional[str] = Query(None)
):
    if ADMIN_TOKEN:
        provided = ""
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        elif x_admin_token:
            provided = x_admin_token.strip()
        elif token:
            provided = token.strip()

        if provided != ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid ADMIN_TOKEN")

    reload_data_index()
    return {
        "ok": True,
        "message": "Index reloaded successfully",
        "chunks_per_site": CHUNKS_PER_SITE,
        "total_chunks": len(DOCS),
    }

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(r: Req):
    hits = retrieve(r.message, k=6)

    # If no chunk scores above zero, answer that the info is not available without calling model
    if not hits:
        return {
            "answer": "I do not have information available regarding that topic in the DarGlobal or Wasalt database.",
            "model": "none",
            "sources": [],
        }

    sources = sorted({h["url"] for h in hits if h.get("url")})
    key = get_openrouter_api_key()

    if not key:
        preview = (
            f"[Search Mode - OPENROUTER_API_KEY not set]\n\n"
            f"Found {len(hits)} relevant source chunks across database:\n\n"
            + "\n\n".join(f"• **{h['title']}** ({h['site']}):\n{h['text'][:300]}..." for h in hits[:3])
            + "\n\nTo enable full AI generative synthesis, please configure OPENROUTER_API_KEY."
        )
        return {"answer": preview, "model": "local-search", "sources": sources}

    context_str = "\n\n".join(
        f"[{h.get('site', 'Unknown')}] {h.get('title', '')} ({h.get('url', '')}):\n{h.get('text', '')}"
        for h in hits
    )

    msgs = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}"}]
    msgs += [m.model_dump() for m in r.history[-6:]]
    msgs.append({"role": "user", "content": r.message})

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://real-estate-chatbot-5gwn.onrender.com",
        "X-Title": "DarGlobal Wasalt Assistant",
    }

    last_error = None
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        for model in MODELS:
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": msgs,
                        "temperature": 0.2,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ans = data["choices"][0]["message"]["content"]
                    return {"answer": ans, "model": model, "sources": sources}

                status = resp.status_code
                body = resp.text
                last_error = f"{model}: HTTP {status} - {body}"

                # Try next model on 404, 429, or 5xx
                if status in (404, 429) or 500 <= status < 600:
                    print(f"[Model Fallback] {model} returned {status}, trying next model...")
                    continue
                else:
                    # Non-retriable error (e.g., 400, 401, 403) - stop and return error
                    break
            except Exception as e:
                last_error = f"{model}: {e}"
                print(f"[Model Fallback] {model} raised exception: {e}, trying next model...")
                continue

    raise HTTPException(status_code=502, detail=f"OpenRouter error: {last_error}")
