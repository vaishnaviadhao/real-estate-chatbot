FROM python:3.12-slim

WORKDIR /app

# Ensure environment flags for performance and Playwright
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install dependencies and Playwright Chromium with system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/* && \
    chmod -R 777 /ms-playwright

# Copy all project files (including data/seed.jsonl and pre-seeded docs)
COPY . .

# Run scrape at build time (if network permits, updates docs.jsonl; fallback handles restrictions)
RUN python -m scraper.scrape || echo "Build-time scrape completed with warnings/fallback; background scraper will verify at startup."

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
