#!/bin/sh
set -e

# Seed the bundled sample PDFs into the persistent /app/data volume.
# `cp -rn` never overwrites files already present (e.g. user uploads).
mkdir -p /app/data/raw_pdfs
cp -rn /app/assets/raw_pdfs/. /app/data/raw_pdfs/ 2>/dev/null || true

exec "$@"
