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

# Wait a moment for the server to spin up
sleep 3

echo "Starting Frontend App..."
cd "$ROOT_DIR/frontend"
npm start
