FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Some pinned packages (e.g. mmh3, uvloop) ship no prebuilt wheels for every
# platform and compile from source, so provide a C toolchain plus curl
# (needed to fetch vendored C deps such as libunwind at build time).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sample PDFs are baked in a stable location and seeded into the persistent
# data volume by docker/entrypoint.sh on container start.
COPY data/raw_pdfs /app/assets/raw_pdfs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
