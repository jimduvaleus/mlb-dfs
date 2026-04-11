#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UI_DIR="$REPO_ROOT/ui"
PID_FILE="$REPO_ROOT/.uvicorn.pid"
LOG_FILE="$REPO_ROOT/server.log"
PORT="${MLB_DFS_PORT:-8000}"

# Check for existing running instance
if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Server already running on http://localhost:$PORT (PID $EXISTING_PID)"
        echo "Run scripts/stop_ui.sh first to restart."
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Build React UI if dist/ is missing or sources are newer than the build
if [ ! -f "$UI_DIR/dist/index.html" ] || \
   find "$UI_DIR/src" -newer "$UI_DIR/dist/index.html" 2>/dev/null | grep -q .; then
    echo "Building React UI..."
    cd "$UI_DIR"
    npm run build
    cd "$REPO_ROOT"
fi

# Activate virtualenv and start the server
source "$REPO_ROOT/venv/bin/activate"

echo "Starting MLB DFS Optimizer UI on http://localhost:$PORT"
uvicorn src.api.server:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-level warning >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait briefly and confirm it started
sleep 1
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server started (PID $SERVER_PID)"
    echo "Open http://localhost:$PORT in your browser"
    echo "Logs: $LOG_FILE"
else
    echo "ERROR: Server failed to start. Check $LOG_FILE for details."
    rm -f "$PID_FILE"
    exit 1
fi
