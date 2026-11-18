"""Utility script to fetch web pages from a list of URLs, extract clean content,
chunk it (800 chars with 100 overlap), and append to data/seed.jsonl.

Usage:
    python scripts/add_seed.py https://darglobal.co.uk/about https://wasalt.sa/about-us
    python scripts/add_seed.py --file urls.txt
"""
import argparse, json, os, re, sys
from pathlib import Path
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

SEED_FILE = Path(os.getenv("SEED_FILE", "data/seed.jsonl"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def determine_site(url: str, custom_site: str = "") -> str:
    if custom_site:
        return custom_site
    host = urlparse(url).netloc.lower()
    if "darglobal" in host:
        return "DarGlobal"
    if "wasalt" in host:
        return "Wasalt"
    return "Unknown"

def clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else "Untitled"
    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) > 20 and (not parts or parts[-1] != t):
            parts.append(t)
    return title, "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
            else:
                last_space = text.rfind(" ", start, end)
                if last_space > start + chunk_size // 2:
                    end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def fetch_page_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--single-process"]
        )
        context = browser.new_context(user_agent=UA)
        context.route("**/*.{png,jpg,jpeg,gif,webp,svg,mp4,woff,woff2,ttf,eot}", lambda r: r.abort())
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        # Scroll to load lazy content
        page.evaluate("""async () => {
            let total = 0;
            while (total < 3000 && total < document.body.scrollHeight) {
                window.scrollBy(0, 400);
                total += 400;
                await new Promise(r => setTimeout(r, 150));
            }
        }""")
        page.wait_for_timeout(1000)
        html = page.content()
        browser.close()
        return html

def fetch_page_httpx(url: str) -> str:
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    }
    with httpx.Client(headers=headers, follow_redirects=True, timeout=25) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text

def fetch_page(url: str) -> str:
    try:
        return fetch_page_playwright(url)
    except Exception as e:
        print(f"[Playwright fallback for {url}]: {e}")
        return fetch_page_httpx(url)

def add_urls_to_seed(urls: list[str], custom_site: str = ""):
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if SEED_FILE.exists():
        for line in SEED_FILE.open(encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                existing_keys.add((d.get("url"), d.get("text", "")[:80]))

    new_chunks = 0
    with SEED_FILE.open("a", encoding="utf-8") as f:
        for url in urls:
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            site = determine_site(url, custom_site)
            print(f"Fetching [{site}] {url}...")
            try:
                html = fetch_page(url)
                title, text = clean_html(html)
                chunks = chunk_text(text)
                for ch in chunks:
                    key = (url, ch[:80])
                    if key not in existing_keys:
                        f.write(json.dumps({
                            "site": site,
                            "url": url,
                            "title": title,
                            "text": ch,
                        }, ensure_ascii=False) + "\n")
                        existing_keys.add(key)
                        new_chunks += 1
                print(f"  -> Added {len(chunks)} chunks for {url}")
            except Exception as e:
                print(f"  -> Error fetching {url}: {e}")

    print(f"\nDone! Added {new_chunks} new chunks to {SEED_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Add URLs to data/seed.jsonl")
    parser.add_argument("urls", nargs="*", help="List of URLs to scrape and add")
    parser.add_argument("--file", "-f", help="Text file containing one URL per line")
    parser.add_argument("--site", "-s", default="", help="Override site name (DarGlobal or Wasalt)")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.file:
        p = Path(args.file)
        if p.exists():
            urls.extend(p.read_text(encoding="utf-8").splitlines())
        else:
            print(f"File not found: {args.file}")
            sys.exit(1)

    if not urls:
        print("No URLs provided. Usage: python scripts/add_seed.py <url1> <url2> ... or --file urls.txt")
        sys.exit(1)

    add_urls_to_seed(urls, custom_site=args.site)

if __name__ == "__main__":
    main()
