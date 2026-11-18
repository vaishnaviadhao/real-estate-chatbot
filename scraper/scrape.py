"""Polite scraper for public pages of DarGlobal and Wasalt.
Output: data/docs.jsonl (one chunk per line: site, url, title, text)."""
import json, os, re, time, sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import httpx
from bs4 import BeautifulSoup

SITES = {
    "DarGlobal": os.getenv("DARGLOBAL_URL", "https://darglobal.co.uk"),
    "Wasalt": os.getenv("WASALT_URL", "https://wasalt.sa"),
}
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_SITE", "120"))
OUT = Path(os.getenv("DATA_FILE", "data/docs.jsonl"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
SKIP = re.compile(r"\.(pdf|jpg|jpeg|png|gif|svg|webp|zip|mp4|css|js|xml)(\?|$)", re.I)

def robots(client, base):
    rp = RobotFileParser()
    try:
        r = client.get(urljoin(base, "/robots.txt"), follow_redirects=True, timeout=15)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
            return rp
    except Exception as e:
        print(f"[{base}] Could not read robots.txt: {e}")
    # Default permissive if robots.txt unavailable
    rp.parse(["User-agent: *", "Allow: /"])
    return rp

def sitemap_urls(client, base, limit=400):
    found, queue, seen = [], [urljoin(base, "/sitemap.xml")], set()
    while queue and len(found) < limit:
        u = queue.pop(0)
        if u in seen: continue
        seen.add(u)
        try:
            r = client.get(u, follow_redirects=True, timeout=15)
            if r.status_code != 200: continue
        except Exception:
            continue
        for l in re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text):
            (queue if l.endswith(".xml") else found).append(l.strip())
    return found

def clean(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        t.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "span", "div"]):
        if el.name in ("span", "div") and el.find(["p", "li", "div", "h1", "h2", "h3"]):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) > 25 and (not parts or parts[-1] != t):
            parts.append(t)
    return title, soup, "\n".join(parts)

def chunk(text, size=900):
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > size and cur:
            out.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip(): out.append(cur)
    return out

def crawl(name, base):
    host = urlparse(base).netloc.replace("www.", "")
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    }
    rows = []
    with httpx.Client(headers=headers, follow_redirects=True, timeout=25) as c:
        rp = robots(c, base)
        queue = [base] + sitemap_urls(c, base)
        seen = set()
        while queue and len(seen) < MAX_PAGES:
            u = queue.pop(0).split("#")[0]
            if u in seen or SKIP.search(u): continue
            curr_host = urlparse(u).netloc.replace("www.", "")
            if curr_host and curr_host != host: continue
            if not rp.can_fetch(UA, u): continue
            seen.add(u)
            try:
                r = c.get(u)
                if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                    continue
                # Check for anti-bot / WAF challenge pages
                if "Incapsula" in r.text or "challenges.cloudflare.com" in r.text:
                    print(f"[{name}] Bot protection challenge encountered at {u}")
                    continue
            except Exception as e:
                print(f"[{name}] Skip {u}: {e}")
                continue

            title, soup, text = clean(r.text)
            if len(text.strip()) > 50:
                for ch in chunk(text):
                    rows.append({"site": name, "url": u, "title": title, "text": ch})
                for a in soup.find_all("a", href=True):
                    next_url = urljoin(u, a["href"])
                    if urlparse(next_url).netloc.replace("www.", "") == host:
                        queue.append(next_url)
            print(f"[{name}] {len(seen)} {u} (chunks: {len(rows)})")
            time.sleep(0.5)
    return rows

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allrows = []
    for n, b in SITES.items():
        allrows += crawl(n, b)

    if allrows:
        with OUT.open("w", encoding="utf-8") as f:
            for r in allrows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Scraped {len(allrows)} live chunks saved to {OUT}")
    else:
        print("Live scrape returned 0 chunks (likely blocked by WAF/anti-bot). Preserving existing dataset if present.")
        if not OUT.exists() or OUT.stat().st_size == 0:
            print("No existing dataset found. Please generate or provide data/docs.jsonl.")
            sys.exit(1)
        else:
            print(f"Preserving existing dataset at {OUT} ({OUT.stat().st_size} bytes).")
