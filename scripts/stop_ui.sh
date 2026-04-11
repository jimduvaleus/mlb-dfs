#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$REPO_ROOT/.uvicorn.pid"
LOG_FILE="$REPO_ROOT/server.log"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE — server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped MLB DFS Optimizer UI (PID $PID)"
    echo "Logs saved to $LOG_FILE"
else
    echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
