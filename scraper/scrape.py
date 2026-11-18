"""Production-grade polite scraper for DarGlobal and Wasalt.
Features:
- Playwright (headless Chromium) with lazy-scroll & networkidle
- Automatic fallback to httpx + BeautifulSoup if Playwright fails
- Separate crawls for DarGlobal and Wasalt seeded with sitemaps and key sections
- Internal link following up to 150 pages with 1s delay and robots.txt checks
- 800-character chunks with 100-character overlap
- Per-site logging and minimum 20-chunk threshold check
"""
import json, os, re, sys, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import httpx
from bs4 import BeautifulSoup

SITES = {
    "DarGlobal": {
        "base": os.getenv("DARGLOBAL_URL", "https://darglobal.co.uk"),
        "seeds": [
            "https://darglobal.co.uk",
            "https://darglobal.co.uk/about",
            "https://darglobal.co.uk/about-us",
            "https://darglobal.co.uk/projects",
            "https://darglobal.co.uk/developments",
            "https://darglobal.co.uk/properties",
            "https://darglobal.co.uk/listings",
            "https://darglobal.co.uk/news",
            "https://darglobal.co.uk/media",
            "https://darglobal.co.uk/contact",
            "https://darglobal.co.uk/contact-us",
            "https://darglobal.co.uk/faq",
            "https://darglobal.co.uk/investor-relations",
        ],
        "sitemap_candidates": [
            "https://darglobal.co.uk/sitemap.xml",
            "https://darglobal.co.uk/sitemap_index.xml",
        ],
    },
    "Wasalt": {
        "base": os.getenv("WASALT_URL", "https://wasalt.sa"),
        "seeds": [
            "https://wasalt.sa",
            "https://wasalt.sa/en",
            "https://wasalt.sa/ar",
            "https://wasalt.sa/about-us",
            "https://wasalt.sa/projects",
            "https://wasalt.sa/developments",
            "https://wasalt.sa/properties",
            "https://wasalt.sa/listings",
            "https://wasalt.sa/news",
            "https://wasalt.sa/contact",
            "https://wasalt.sa/contact-us",
            "https://wasalt.sa/faq",
            "https://wasalt.sa/terms",
        ],
        "sitemap_candidates": [
            "https://wasalt.sa/sitemap.xml",
            "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
        ],
    },
}

MAX_PAGES = int(os.getenv("MAX_PAGES_PER_SITE", "150"))
DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "1.0"))
OUT_FILE = Path(os.getenv("DATA_FILE", "data/docs.jsonl"))
MIN_CHUNKS_PER_SITE = int(os.getenv("MIN_CHUNKS_PER_SITE", "20"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
SKIP_EXT = re.compile(r"\.(pdf|jpg|jpeg|png|gif|svg|webp|zip|mp4|avi|mp3|css|js|xml|json)(\?|$)", re.I)

def get_robots_parser(client: httpx.Client, base_url: str) -> RobotFileParser:
    rp = RobotFileParser()
    try:
        r = client.get(urljoin(base_url, "/robots.txt"), follow_redirects=True, timeout=15)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
            return rp
    except Exception as e:
        print(f"[{base_url}] robots.txt not accessible: {e}")
    rp.parse(["User-agent: *", "Allow: /"])
    return rp

def fetch_sitemap_urls(client: httpx.Client, candidates: list[str], limit: int = 200) -> list[str]:
    urls = []
    seen = set()
    for sitemap_url in candidates:
        if len(urls) >= limit:
            break
        try:
            r = client.get(sitemap_url, follow_redirects=True, timeout=15)
            if r.status_code != 200:
                continue
            text = r.text
            # Simple XML loc extraction
            locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", text)
            for loc in locs:
                loc = loc.strip()
                if loc and loc not in seen and not SKIP_EXT.search(loc):
                    seen.add(loc)
                    if not loc.endswith(".xml") and not loc.endswith(".gz"):
                        urls.append(loc)
                if len(urls) >= limit:
                    break
        except Exception as e:
            print(f"Could not fetch sitemap {sitemap_url}: {e}")
    return urls

def clean_html(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    for el in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        el.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    parts = []
    # Extract visible text from headings, paragraphs, list items, tables
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(text) > 20 and (not parts or parts[-1] != text):
            parts.append(text)

    # Collect internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith("javascript:") and not href.startswith("mailto:") and not href.startswith("tel:"):
            links.append(href)

    return title, parts, links

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
            # Prefer splitting at a sentence or space boundary
            last_period = text.rfind(". ", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
            else:
                last_space = text.rfind(" ", start, end)
                if last_space > start + chunk_size // 2:
                    end = last_space
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

class PageFetcher:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.playwright_available = False
        self._init_playwright()

    def _init_playwright(self):
        try:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-zygote",
                    "--single-process",
                    "--disable-extensions",
                ]
            )
            self.context = self.browser.new_context(
                user_agent=UA,
                viewport={"width": 1280, "height": 800},
            )
            # Abort heavy assets to keep memory under 150MB
            self.context.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,mp4,avi,woff,woff2,ttf,eot}",
                lambda route: route.abort()
            )
            self.playwright_available = True
            print("[Fetcher] Playwright Chromium initialized successfully.")
        except Exception as e:
            print(f"[Fetcher] Playwright unavailable ({e}). Using httpx + BeautifulSoup fallback.")
            self.playwright_available = False

    def fetch(self, url: str, client: httpx.Client) -> str:
        if self.playwright_available:
            try:
                page = self.context.new_page()
                page.goto(url, wait_until="networkidle", timeout=25000)
                # Scroll down to load lazy content
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
                page.close()
                return html
            except Exception as e:
                print(f"[Playwright error for {url}]: {e}. Falling back to httpx.")
                # Fall through to httpx

        # httpx fallback
        resp = client.get(url, follow_redirects=True, timeout=20)
        resp.raise_for_status()
        return resp.text

    def close(self):
        if self.context:
            try: self.context.close()
            except Exception: pass
        if self.browser:
            try: self.browser.close()
            except Exception: pass
        if self.playwright:
            try: self.playwright.stop()
            except Exception: pass

def crawl_site(site_name: str, site_config: dict, fetcher: PageFetcher) -> tuple[int, list[dict]]:
    base_url = site_config["base"]
    parsed_base = urlparse(base_url)
    host = parsed_base.netloc.replace("www.", "").lower()

    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    }

    chunks = []
    seen = set()
    queue = list(site_config["seeds"])

    with httpx.Client(headers=headers, follow_redirects=True, timeout=25) as client:
        rp = get_robots_parser(client, base_url)
        sitemap_urls = fetch_sitemap_urls(client, site_config.get("sitemap_candidates", []))
        queue.extend(sitemap_urls)

        # De-duplicate queue
        dedup_queue = []
        for u in queue:
            clean_u = u.split("#")[0].strip()
            if clean_u and clean_u not in seen and clean_u not in dedup_queue:
                dedup_queue.append(clean_u)
        queue = dedup_queue

        print(f"\n[{site_name}] Starting crawl with {len(queue)} seed URLs. Max pages: {MAX_PAGES}")

        pages_crawled = 0
        while queue and pages_crawled < MAX_PAGES:
            u = queue.pop(0)
            if u in seen or SKIP_EXT.search(u):
                continue

            parsed_u = urlparse(u)
            curr_host = parsed_u.netloc.replace("www.", "").lower()
            if curr_host and curr_host != host:
                continue

            if not rp.can_fetch(UA, u):
                print(f"[{site_name}] Disallowed by robots.txt: {u}")
                continue

            seen.add(u)
            pages_crawled += 1

            try:
                html = fetcher.fetch(u, client)
                # Verify page is not a bot-block challenge page
                if "Incapsula" in html or "challenges.cloudflare.com" in html:
                    print(f"[{site_name}] Bot protection challenge encountered at {u}")
                    continue

                title, parts, links = clean_html(html)
                full_text = "\n".join(parts)
                if len(full_text.strip()) > 60:
                    text_chunks = chunk_text(full_text)
                    for ch in text_chunks:
                        chunks.append({
                            "site": site_name,
                            "url": u,
                            "title": title or site_name,
                            "text": ch,
                        })

                # Follow internal links
                for href in links:
                    next_url = urljoin(u, href).split("#")[0]
                    next_host = urlparse(next_url).netloc.replace("www.", "").lower()
                    if next_host == host and next_url not in seen and next_url not in queue and not SKIP_EXT.search(next_url):
                        queue.append(next_url)

                print(f"[{site_name}] [{pages_crawled}/{MAX_PAGES}] {u} -> Chunks: {len(chunks)}")
            except Exception as e:
                print(f"[{site_name}] Error crawling {u}: {e}")

            time.sleep(DELAY_SECONDS)

    return pages_crawled, chunks

def main():
    print("=" * 60)
    print("STARTING CRAWLER FOR DARGLOBAL AND WASALT")
    print("=" * 60)

    fetcher = PageFetcher()
    all_chunks = []
    stats = {}

    try:
        for site_name, config in SITES.items():
            pages_visited, site_chunks = crawl_site(site_name, config, fetcher)
            stats[site_name] = {"pages": pages_visited, "chunks": len(site_chunks)}
            all_chunks.extend(site_chunks)
    finally:
        fetcher.close()

    print("\n" + "=" * 60)
    print("=== SCRAPE SUMMARY ===")
    print("=" * 60)
    for site, s in stats.items():
        print(f"[{site}] Pages visited: {s['pages']}, Chunks extracted: {s['chunks']}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if all_chunks:
        with OUT_FILE.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nSuccessfully wrote {len(all_chunks)} chunks to {OUT_FILE}")

    # Enforce threshold requirement: Exit with error if either site has fewer than 20 chunks
    darglobal_chunks = stats.get("DarGlobal", {}).get("chunks", 0)
    wasalt_chunks = stats.get("Wasalt", {}).get("chunks", 0)

    if darglobal_chunks < MIN_CHUNKS_PER_SITE or wasalt_chunks < MIN_CHUNKS_PER_SITE:
        print(
            f"\n[ERROR] Minimum chunk threshold ({MIN_CHUNKS_PER_SITE}) not met! "
            f"DarGlobal: {darglobal_chunks}, Wasalt: {wasalt_chunks}",
            file=sys.stderr
        )
        sys.exit(1)

    print("\nScraping completed successfully within thresholds.")
    sys.exit(0)

if __name__ == "__main__":
    main()
