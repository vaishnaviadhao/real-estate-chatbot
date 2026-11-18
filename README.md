---
title: DarGlobal & Wasalt Real Estate Assistant
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# DarGlobal and Wasalt Real Estate AI Assistant

An intelligent real estate AI assistant built using Retrieval-Augmented Generation (RAG) over public property listings and corporate portfolios from **DarGlobal** and **Wasalt**.

## System Architecture

- **Data Layer:** Web scrapers (`httpx` + `BeautifulSoup`) and curated dataset (`data/docs.jsonl`) covering DarGlobal luxury developments (Dubai, London, Oman, Qatar, Spain) and Wasalt Saudi real estate marketplace (Riyadh, Jeddah, Dammam).
- **Retrieval Engine:** BM25 lexical ranking (`rank-bm25`) with dual-script tokenization (English and Arabic).
- **LLM Synthesis:** OpenRouter free tier models (`meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `mistralai/mistral-small-24b-instruct-2501:free`, `qwen/qwen-2.5-72b-instruct:free`) with multi-model fallback.
- **Backend API:** FastAPI with async HTTP client.
- **Frontend:** Lightweight responsive chat UI (`static/index.html`).
- **Containerization:** Docker container with non-root security compliance.

---

## Running Locally

### 1. Direct Python
```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Export OpenRouter API Key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 2. Docker Compose
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
docker compose up --build
```

---

## Deployment Options

### Option A: Hugging Face Spaces (Free & Recommended)
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Docker** as Space SDK (Blank template).
3. Under **Settings > Variables and secrets**, add secret `OPENROUTER_API_KEY`.
4. Push this repository to your Hugging Face Space git remote:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
   git add .
   git commit -m "Deploy DarGlobal and Wasalt Chatbot"
   git push -u origin main
   ```
5. Your chatbot will be live at `https://<YOUR_USERNAME>-<SPACE_NAME>.hf.space`.

### Option B: Render (Free Web Service)
1. Push this repository to GitHub.
2. In [Render](https://dashboard.render.com/), create a **New Web Service** pointing to the GitHub repo.
3. Select **Docker** environment.
4. Add environment variable `OPENROUTER_API_KEY`.
5. Click **Deploy**.

---

## API Endpoints

- `GET /`: Interactive web chat interface.
- `GET /health`: Healthcheck, total indexed document chunks, and model configuration.
- `POST /chat`: Chat query endpoint accepting `{"message": "string", "history": []}`.
