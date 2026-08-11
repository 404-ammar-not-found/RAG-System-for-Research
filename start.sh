#!/usr/bin/env bash
# One command: deps, database, API, UI. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")"

python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' ||
  { echo "Python 3.12+ required (found $(python3 -V))."; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — add an API key (Gemini, Anthropic or OpenAI), then re-run."
  echo "Free Gemini key: https://aistudio.google.com/apikey"
  exit 1
fi
grep -Eq '^(GEMINI|GOOGLE|ANTHROPIC|OPENAI)_API_KEY *= *\S' .env ||
  { echo "Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY in .env, then re-run."; exit 1; }
# Anthropic publishes no embedding endpoint, so vectors come from elsewhere —
# on-device by default, which needs no key and no quota.
if grep -Eq '^ANTHROPIC_API_KEY *= *\S' .env &&
   ! grep -Eq '^(GEMINI|GOOGLE|OPENAI)_API_KEY *= *\S' .env; then
  echo "Anthropic has no embedding API — embedding locally (all-MiniLM-L6-v2, ~79MB on first run)."
fi

[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q -r requirements.txt

# FalkorDB holds the knowledge graph. Without it the system still answers from
# passages alone, so a missing container is a warning, not a failure.
# Port 6380 because macOS dev boxes often already run redis-server on 6379.
if command -v docker >/dev/null; then
  docker start falkordb >/dev/null 2>&1 ||
    docker run -d --name falkordb -p 6380:6379 -p 3000:3000 \
      -v "$PWD/falkor_data:/var/lib/falkordb/data" falkordb/falkordb:latest >/dev/null
else
  echo "[WARN] docker not found — running passage-only, no knowledge graph."
fi

[ -d web-visualizer/node_modules ] || npm --prefix web-visualizer install

trap 'kill 0' EXIT
./venv/bin/uvicorn src.api.web_api:app --port 8000 &
npm --prefix web-visualizer run dev &

echo
echo "  UI            http://localhost:5173"
echo "  API           http://localhost:8000/docs"
echo "  Graph browser http://localhost:3000"
echo
wait
