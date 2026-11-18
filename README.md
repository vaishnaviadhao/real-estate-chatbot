# DarGlobal and Wasalt Real Estate AI Chatbot

A production-ready real estate AI assistant built using Retrieval-Augmented Generation (RAG) over public property listings and corporate portfolios from **DarGlobal** and **Wasalt**.

## System Architecture

- **Scraper Engine (`scraper/scrape.py`):**
  - Uses **Playwright (headless Chromium)** with lazy-scroll & `networkidle` to render JavaScript-heavy single-page applications.
  - Automatic fallback to **httpx + BeautifulSoup** if Playwright is unavailable.
  - Separate crawls for DarGlobal (`darglobal.co.uk`) and Wasalt (`wasalt.sa`) seeded with homepages, `sitemap.xml`, and key sections (`about`, `projects`, `developments`, `properties`, `listings`, `news`, `contact`, `faq`).
  - Internal link following up to 150 pages with a 1-second delay, respecting `robots.txt`.
  - Content chunking into ~800-character segments with 100-character overlap.
  - Per-site logging with minimum 20-chunk threshold validation.
- **Seed Data Fallback (`data/seed.jsonl` & `scripts/add_seed.py`):**
  - Application merges `data/docs.jsonl` and `data/seed.jsonl` at startup and reload.
  - `scripts/add_seed.py` allows manually adding pages from a list of URLs if automated scraping is thin.
- **Retrieval Engine (`app/main.py`):**
  - BM25 Okapi with dual-script tokenization (Arabic & English).
  - Site-aware query filtering: boosts DarGlobal chunks when the query mentions DarGlobal; boosts Wasalt chunks when Wasalt is mentioned.
  - Balanced top-6 retrieval across both sites when querying general topics.
  - Zero-match safeguard: answers directly that information is not available rather than calling LLMs with empty context.
- **Model Orchestration:**
  - Multi-model fallback across OpenRouter free models on 404, 429, or 5xx errors.
  - Models: `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `qwen/qwen-2.5-72b-instruct:free`, `mistralai/mistral-small-24b-instruct-2501:free`, `google/gemma-3-27b-it:free`.
- **Containerization (`Dockerfile`):**
  - Optimized for Render's free tier (512 MB RAM).
  - Runs build-time scrape and automatic background scraping at startup if `docs.jsonl` has fewer than 100 chunks.
  - Dynamic `$PORT` binding.

---

## API Endpoints

- `GET /`: Interactive web chat interface.
- `GET /health`: Returns chunk counts per site, total chunks, model list, and API key status.
- `POST /reload`: Rebuilds the search index without restarting the server. Protected by `ADMIN_TOKEN`.
- `POST /chat`: AI question-answering endpoint.

---

## Render Environment Variables

Set these in your [Render Dashboard](https://dashboard.render.com/) under **Environment**:

| Variable | Required | Description | Example |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** | Your API key from [OpenRouter](https://openrouter.ai/) | `sk-or-v1-xxxxxxxx` |
| `ADMIN_TOKEN` | Optional | Token to secure the `/reload` endpoint | `my-secure-admin-token` |
| `OPENROUTER_MODELS` | Optional | Comma-separated list of fallback models | `meta-llama/llama-3.3-70b-instruct:free,google/gemini-2.0-flash-exp:free` |
| `MAX_PAGES_PER_SITE`| Optional | Max pages to crawl per site (default: 150) | `150` |
| `CRAWL_DELAY_SECONDS` | Optional | Delay between requests in seconds (default: 1.0) | `1.0` |

---

## Manual Seed Utility

To add URLs directly to `data/seed.jsonl`:
```bash
python scripts/add_seed.py https://darglobal.co.uk/about https://wasalt.sa/about-us
# Or from a file:
python scripts/add_seed.py --file urls.txt
```
Then trigger `/reload` on your deployed app:
```bash
curl -X POST "https://<your-render-url>/reload" -H "Authorization: Bearer <ADMIN_TOKEN>"
```
