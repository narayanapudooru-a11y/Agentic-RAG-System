#!/usr/bin/env bash
set -euo pipefail

# test runner for the repo's test-app skill
# Exits non-zero on any failure

# Ensure OPENAI_API_KEY is set
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY must be set in the environment"
  exit 2
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running unit tests..."
pytest -q

# If there is no streamlit app, fail the smoke test (this is intentional so UI PRs must add the app)
APP_PATH=src/webui/app.py
if [ ! -f "$APP_PATH" ]; then
  echo "WARNING: Streamlit app not found at $APP_PATH — smoke test will fail by design for UI-less commits"
  exit 1
fi

# Start streamlit in background on port 8501
echo "Starting Streamlit app..."
streamlit run "$APP_PATH" --server.port 8501 > /tmp/streamlit.test.log 2>&1 &
STREAMPID=$!

# Give it some time to boot
sleep 5

# Smoke test: fetch root page
echo "Performing HTTP smoke check..."
if curl -sS --fail http://localhost:8501/ > /dev/null; then
  echo "Smoke check OK"
  kill $STREAMPID || true
  wait $STREAMPID 2>/dev/null || true
  echo "All tests passed"
  exit 0
else
  echo "Smoke check failed. Check /tmp/streamlit.test.log for details"
  kill $STREAMPID || true
  wait $STREAMPID 2>/dev/null || true
  exit 3
fi
