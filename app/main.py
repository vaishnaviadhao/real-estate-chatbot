import json, os, re, subprocess, sys
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

DATA = Path(os.getenv("DATA_FILE", "data/docs.jsonl"))
KEY = os.getenv("OPENROUTER_API_KEY", "")
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "meta-llama/llama-3.3-70b-instruct:free,google/gemini-2.0-flash-exp:free,qwen/qwen-2.5-72b-instruct:free,mistralai/mistral-small-24b-instruct-2501:free,google/gemma-3-27b-it:free",
).split(",") if m.strip()]

SYSTEM = (
    "You are an expert real estate AI assistant specialized in DarGlobal and Wasalt.\n"
    "Answer the user's question accurately using ONLY the retrieved context below.\n"
    "If the answer is not present in the context, state politely that you do not have that specific information.\n"
    "Be concise, informative, and specify whether you are referring to DarGlobal or Wasalt."
)

def tok(s: str) -> list[str]:
    # Match alphanumeric words in Latin and Arabic scripts
    return re.findall(r"[\w\u0600-\u06FF]+", s.lower())

if not DATA.exists() or DATA.stat().st_size == 0:
    print(f"Data file {DATA} missing or empty. Running scraper...")
    subprocess.run([sys.executable, "-m", "scraper.scrape"], check=False)

DOCS = [json.loads(line) for line in DATA.open(encoding="utf-8")] if DATA.exists() and DATA.stat().st_size > 0 else []
BM = BM25Okapi([tok(d["title"] + " " + d["text"]) for d in DOCS]) if DOCS else None

app = FastAPI(title="DarGlobal & Wasalt Real Estate Assistant")

class Msg(BaseModel):
    role: str
    content: str

class Req(BaseModel):
    message: str
    history: list[Msg] = []

def retrieve(query: str, k: int = 5):
    if not BM or not DOCS:
        return []
    tokens = tok(query)
    if not tokens:
        return []
    scores = BM.get_scores(tokens)
    top_indices = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)[:k]
    return [DOCS[i] for i in top_indices if scores[i] > 0]

@app.get("/health")
def health():
    return {
        "ok": True,
        "chunks": len(DOCS),
        "models": MODELS,
        "has_api_key": bool(KEY),
    }

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(r: Req):
    hits = retrieve(r.message)
    sources = sorted({h["url"] for h in hits})

    if not KEY:
        if hits:
            preview = (
                f"[Search Mode - OPENROUTER_API_KEY not configured]\n\n"
                f"Retrieved {len(hits)} relevant source documents from the database:\n\n"
                + "\n\n".join(f"• **{h['title']}** ({h['site']}):\n{h['text'][:350]}..." for h in hits[:3])
                + "\n\nTo enable full AI generative synthesis, set your OPENROUTER_API_KEY environment variable."
            )
            return {"answer": preview, "model": "local-retrieval", "sources": sources}
        else:
            return {
                "answer": (
                    "No direct matching documents found in the database. "
                    "Please set OPENROUTER_API_KEY to enable LLM generation."
                ),
                "model": "local-retrieval",
                "sources": [],
            }

    ctx = "\n\n".join(f"[{h['site']}] {h['title']} ({h['url']}):\n{h['text']}" for h in hits) or "No relevant context found in database."
    msgs = [{"role": "system", "content": f"{SYSTEM}\n\nContext:\n{ctx}"}]
    msgs += [m.model_dump() for m in r.history[-6:]]
    msgs.append({"role": "user", "content": r.message})

    err = None
    headers = {
        "Authorization": f"Bearer {KEY}",
        "HTTP-Referer": "https://github.com/vaishnavi/re-chatbot",
        "X-Title": "DarGlobal and Wasalt Real Estate Assistant",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
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
                err = f"{model}: HTTP {resp.status_code} - {resp.text[:200]}"
            except Exception as e:
                err = f"{model}: {e}"

    raise HTTPException(status_code=502, detail=f"OpenRouter models failed: {err}")
