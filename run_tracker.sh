#!/bin/bash
#
# Launcher script for the Apple Notes Live Tracker
# Runs the tracker in the background and manages its lifecycle
#

set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_SCRIPT="$SCRIPT_DIR/server/tracker.ts"

# Check if tracker script exists
if [ ! -f "$TRACKER_SCRIPT" ]; then
    echo "❌ Error: tracker.ts not found at $TRACKER_SCRIPT"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Apple Notes Live Tracker Launcher"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the tracker in the background"
    echo "  stop      - Stop the running tracker"
    echo "  restart   - Restart the tracker"
    echo "  status    - Check if tracker is running"
    echo "  run       - Run the tracker in the foreground"
    echo ""
}

# Get tracker PID file
PID_FILE="$SCRIPT_DIR/tracker.pid"

# Check if tracker is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Start the tracker
start_tracker() {
    if is_running; then
        echo "⚠️  Tracker is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    echo "🚀 Starting Apple Notes Live Tracker..."
    
    # Start tracker in background
    cd "$SCRIPT_DIR"
    # Use ts-node or bun to run TypeScript
    if command -v ts-node &> /dev/null; then
        ts-node "$TRACKER_SCRIPT" > "$SCRIPT_DIR/tracker.log" 2>&1 &
    elif command -v bun &> /dev/null; then
        bun "$TRACKER_SCRIPT" > "$SCRIPT_DIR/tracker.log" 2>&1 &
    else
        echo "⚠️  No TypeScript runner found (ts-node, bun)"
        echo "Install with: npm install -g ts-node"
        exit 1
    fi
    local pid=$!
    
    # Save PID
    echo "$pid" > "$PID_FILE"
    
    echo "✅ Tracker started with PID: $pid"
    echo "📝 Logs available at: $SCRIPT_DIR/tracker.log"
    echo ""
    echo "The tracker will check for open notes every 10 seconds."
    echo "Press Ctrl+C in this terminal to stop the tracker."
}

# Stop the tracker
stop_tracker() {
    if ! is_running; then
        echo "ℹ️  Tracker is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    echo "⏹️  Stopping tracker (PID: $pid)..."
    
    kill "$pid" 2>/dev/null || true
    
    # Wait for process to terminate
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        echo "⚠️  Force killing tracker..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    
    # Remove PID file
    rm -f "$PID_FILE"
    
    echo "✅ Tracker stopped"
}

# Show status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "✅ Tracker is running"
        echo "   PID: $pid"
        echo "   Logs: $SCRIPT_DIR/tracker.log"
        
        # Show recent logs if available
        if [ -f "$SCRIPT_DIR/tracker.log" ]; then
            echo ""
            echo "📝 Recent logs:"
            tail -n 5 "$SCRIPT_DIR/tracker.log"
        fi
    else
        echo "❌ Tracker is not running"
        echo ""
        echo "Start the tracker with: $0 start"
    fi
}

# Run tracker in foreground
run_tracker() {
    echo "🚀 Starting Apple Notes Live Tracker (foreground mode)"
    echo "Press Ctrl+C to stop"
    echo ""
    
    cd "$SCRIPT_DIR"
    # Use ts-node or tsc to run TypeScript
    if command -v ts-node &> /dev/null; then
        ts-node "$TRACKER_SCRIPT"
    elif command -v bun &> /dev/null; then
        bun "$TRACKER_SCRIPT"
    else
        echo "⚠️ No TypeScript runner found (ts-node, bun)"
        echo "Install with: npm install -g ts-node"
        exit 1
    fi
}

# Main command handler
case "${1:-}" in
    start)
        start_tracker
        ;;
    stop)
        stop_tracker
        ;;
    restart)
        stop_tracker
        start_tracker
        ;;
    status)
        show_status
        ;;
    run)
        run_tracker
        ;;
    *)
        show_usage
        ;;
esac
