FROM python:3.11-slim

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create database directories so volume mounts land in the right place
RUN mkdir -p database/chroma_db

# Optional: Chromium for Playwright-backed scrapers (TechCrunch, etc.).
# Uncomment the next two lines to bake browsers into the image (~400MB+).
# Default: omit — run `playwright install chromium` on the host for full scraper support.
# RUN playwright install-deps chromium
# RUN playwright install chromium

# Expose the FastAPI port
EXPOSE 8000

# Start the agent service
CMD ["python", "main.py", "serve", "--port", "8000"]
