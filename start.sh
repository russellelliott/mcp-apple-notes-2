#!/bin/bash

# Store the root directory
ROOT_DIR="$(pwd)"

echo "Starting Backend Server..."
# Start the backend server in the background
python "$ROOT_DIR/backend/scripts/server.py" &
BACKEND_PID=$!
echo "Backend Server started with PID $BACKEND_PID"

# Function to cleanup background process on exit
cleanup() {
    echo ""
    echo "Stopping backend server (PID $BACKEND_PID)..."
    kill $BACKEND_PID
    exit
}

# Trap exit signals to ensure cleanup occurs
trap cleanup EXIT INT TERM

# Wait for the backend to report healthy before starting the frontend
echo "Waiting for backend to become healthy..."
BACKEND_URL="http://127.0.0.1:8000/health"
MAX_RETRIES=120
RETRY_INTERVAL=1
retry=0

while true; do
    # If backend process died, abort
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Backend process $BACKEND_PID exited unexpectedly. Check logs." >&2
        exit 1
    fi

    # Try to hit health endpoint
    resp=$(curl -s --fail --max-time 2 "$BACKEND_URL" 2>/dev/null || true)
    if [ -n "$resp" ] && echo "$resp" | grep -q '"status"\s*:\s*"ok"'; then
        echo "Backend healthy."
        break
    fi

    retry=$((retry + 1))
    if [ "$retry" -ge "$MAX_RETRIES" ]; then
        echo "Timed out waiting for backend to become healthy after $((MAX_RETRIES * RETRY_INTERVAL))s" >&2
        kill "$BACKEND_PID" 2>/dev/null || true
        exit 1
    fi

    sleep "$RETRY_INTERVAL"
done

echo "Starting Frontend App..."
cd "$ROOT_DIR/frontend"
npm start
