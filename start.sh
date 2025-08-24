#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env || true
echo "Edit .env to add your POLYGON_API_KEY"
echo "Starting server at http://127.0.0.1:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
