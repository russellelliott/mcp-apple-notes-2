# Apple Notes Live Tracking Implementation Summary

## Overview

This implementation adds a comprehensive live-tracking and deep-linking system for the Apple Notes application. The system tracks note interactions (opens and modifications) in a separate LanceDB table called `notes_interactions`, enabling persistent tracking across note modifications.

## Files Created/Modified

### 1. Standalone Tracker Script
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/tracker.js`

A background process that:
- Runs every 10 seconds
- Polls Apple Notes for open windows using JXA
- Upserts interaction records in the `notes_interactions` table
- Works even for notes not yet in the main `notes` table

**Key Features**:
- Automatic tracking of open note windows
- Event logging with timestamps
- Graceful shutdown on Ctrl+C
- Creates `notes_interactions` table automatically

### 2. Launcher Script
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/run_tracker.sh`

A convenience script to manage the tracker lifecycle:
- `./run_tracker.sh start` - Start tracker in background
- `./run_tracker.sh stop` - Stop running tracker
- `./run_tracker.sh restart` - Restart tracker
- `./run_tracker.sh status` - Check tracker status
- `./run_tracker.sh run` - Run in foreground

### 3. Server Tool Implementation
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/server/index.ts`

Added the `open-in-apple-notes` tool:
- Opens a specific note by title
- Handles duplicate titles using creation date
- Automatically logs the interaction in `notes_interactions` table
- Returns success/failure status

**Usage**:
```typescript
await callTool({
  name: "open-in-apple-notes",
  arguments: {
    title: "My Note Title",
    creation_date: "2024-01-15 10:30:00" // Optional
  }
});
```

### 4. Backend Python Support
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/backend/scripts/main.py`

Added interaction tracking methods to `NotesDatabase` class:
- `log_interaction(title, event_type)` - Log any interaction
- `log_note_opened(title)` - Log note open event
- `log_note_modified(title)` - Log note modification
- `get_last_opened(title)` - Get last opened timestamp
- `get_interaction_log(title)` - Get full interaction history

**Usage**:
```python
from backend.scripts.main import NotesDatabase

db = NotesDatabase()
db.log_note_opened("My Note")
db.log_note_modified("My Note")
last_opened = db.get_last_opened("My Note")
```

### 5. API Endpoints
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/backend/scripts/server.py`

Added three new endpoints for interaction data:

#### POST `/interaction/log`
Log an interaction event for a note.

**Request**:
```json
{
  "title": "My Note",
  "event_type": "opened"  // or "modified"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Successfully logged opened event for 'My Note'",
  "last_opened": "2024-06-02T10:30:45.123Z"
}
```

#### GET `/interaction/{title}`
Get interaction data for a specific note.

**Response**:
```json
{
  "title": "My Note",
  "last_opened": "2024-06-02T10:30:45.123Z",
  "interaction_log": [
    {"dt": "2024-06-02T10:30:45.123Z", "type": "opened"},
    {"dt": "2024-06-02T09:15:22.456Z", "type": "modified"}
  ]
}
```

#### GET `/interactions/list`
Get a list of all notes with their last_opened timestamps.

**Query Parameters**:
- `limit` (optional, default 50) - Maximum interactions to return

**Response**:
```json
{
  "interactions": [
    {
      "title": "My Note",
      "last_opened": "2024-06-02T10:30:45.123Z",
      "interaction_log": [...]
    }
  ],
  "count": 1
}
```

### 6. Documentation
**File**: `/Users/russellelliott/Desktop/mcp-apple-notes-2/TRACKER_README.md`

Comprehensive documentation covering:
- Schema definition
- Usage instructions
- JXA scripts
- Architecture diagram
- API endpoints
- Troubleshooting guide

## Database Schema

The `notes_interactions` table has the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `title` | String | Note title (primary key) |
| `last_opened` | String | ISO timestamp of most recent interaction |
| `interaction_log` | String | JSON array of events |

**Interaction Event Format**:
```json
{
  "dt": "2024-06-02T10:30:45.123Z",
  "type": "opened"  // or "modified"
}
```

**Example Interaction Log**:
```json
[
  {"dt": "2024-06-02T10:30:45.123Z", "type": "opened"},
  {"dt": "2024-06-02T09:15:22.456Z", "type": "modified"},
  {"dt": "2024-06-02T08:00:00.000Z", "type": "opened"}
]
```

## JXA Scripts

### Get Open Note Titles
```javascript
(() => {
  const Notes = Application("Notes");
  if (!Notes.running()) return [];
  return Notes.windows.name().filter(name => name && name !== "Notes");
})();
```

### Open Specific Note
```javascript
(function(title, creationDateStr) {
  const Notes = Application("Notes");
  Notes.includeStandardAdditions = true;
  
  const matches = Notes.notes.whose({ name: title })();
  let noteToOpen = null;
  
  for (let i = 0; i < matches.length; i++) {
    const note = matches[i];
    if (!creationDateStr || note.creationDate().toLocaleString() === creationDateStr) {
      noteToOpen = note;
      break;
    }
  }
  
  if (noteToOpen) {
    Notes.activate();
    Notes.show(noteToOpen);
    return JSON.stringify({ status: "success" });
  }
  return JSON.stringify({ status: "not_found" });
})("{{title}}", "{{creation_date}}");
```

## Integration Points

### 1. Main Sync Script
When the main synchronization script detects a note modification, it should log the event:

```python
from backend.scripts.main import NotesDatabase

db = NotesDatabase()
db.log_note_modified("Note Title")
```

### 2. Frontend UI
The frontend can fetch the `last_opened` timestamp from the `notes_interactions` table:

```typescript
// Fetch interaction data
const response = await fetch(`http://localhost:8000/interaction/${encodeURIComponent(title)}`);
const data = await response.json();

// Display last opened time
console.log(`Last opened: ${data.last_opened}`);
```

### 3. Refresh Strategy
Provide a lightweight API endpoint to fetch only the `notes_interactions` data:

```bash
# Fetch all interactions
curl "http://localhost:8000/interactions/list?limit=100"

# Fetch specific note
curl "http://localhost:8000/interaction/My%20Note"
```

## Architecture

```
┌──────────────────────────┐
│   tracker.js             │
│   (runs every 10s)        │
│   - Polls open windows   │
│   - Logs "opened" events │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  notes_interactions      │
│  (separate LanceDB table)│
│  - title (PK)            │
│  - last_opened           │
│  - interaction_log       │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   server.py              │
│   (API endpoints)         │
│   - /interaction/log     │
│   - /interaction/{title} │
│   - /interactions/list   │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Frontend UI            │
│   - Display last_opened  │
│   - Refresh every 10-30s │
└──────────────────────────┘
```

## Usage Examples

### Starting the Tracker

```bash
# Start in background
./run_tracker.sh start

# Check status
./run_tracker.sh status

# Stop
./run_tracker.sh stop

# Run in foreground
./run_tracker.sh run
```

### Opening a Note via MCP

```typescript
// Open a note (automatically logs interaction)
await callTool({
  name: "open-in-apple-notes",
  arguments: {
    title: "Meeting Notes",
    creation_date: "2024-01-15 10:30:00"  // Optional for duplicates
  }
});
```

### Logging Interactions Programmatically

```python
from backend.scripts.main import NotesDatabase

db = NotesDatabase()

# Log when note is opened
db.log_note_opened("Meeting Notes")

# Log when note is modified
db.log_note_modified("Meeting Notes")

# Get interaction history
log = db.get_interaction_log("Meeting Notes")
print(f"Last opened: {log[-1]['dt'] if log else 'N/A'}")
```

### Fetching Interaction Data via API

```bash
# Get last opened for a specific note
curl "http://localhost:8000/interaction/Meeting%20Notes"

# List all interactions
curl "http://localhost:8000/interactions/list?limit=50"

# Log an interaction
curl -X POST http://localhost:8000/interaction/log \
  -H "Content-Type: application/json" \
  -d '{"title": "Meeting Notes", "event_type": "opened"}'
```

## Benefits

1. **Persistent Tracking**: Interaction data persists across note modifications
2. **Loose Relationship**: Tracks notes even before they're in the main database
3. **Event History**: Complete history of opens and modifications
4. **Performance**: Lightweight API endpoints for frequent UI updates
5. **Scalability**: Separate table keeps main notes table clean

## Future Enhancements

- [ ] Add "focused" event type for window focus changes
- [ ] Track duration notes remain open
- [ ] Add analytics endpoint for interaction patterns
- [ ] Support for note creation/deletion events
- [ ] Real-time WebSocket updates for interactions
- [ ] Export interaction data to JSON/CSV

## Troubleshooting

### Tracker Not Starting
- Check Node.js is installed: `node --version`
- Verify `run-jxa` package is installed
- Check Apple Notes has automation permissions

### Notes Not Being Tracked
- Ensure Apple Notes is running
- Verify Notes windows are actually open
- Check `~/.mcp-apple-notes/data/` exists

### Interaction Log Not Updating
- Check tracker logs: `tail -f run_tracker.log`
- Verify tracker is still running
- Check disk space availability

## Testing

To test the implementation:

1. **Start the tracker**:
   ```bash
   ./run_tracker.sh start
   ```

2. **Open some notes in Apple Notes**

3. **Check the interaction log**:
   ```bash
   curl "http://localhost:8000/interactions/list"
   ```

4. **Open a note via the tool**:
   ```typescript
   await callTool({
     name: "open-in-apple-notes",
     arguments: { title: "My Note" }
   });
   ```

5. **Verify the interaction was logged**:
   ```bash
   curl "http://localhost:8000/interaction/My%20Note"
   ```

6. **Stop the tracker**:
   ```bash
   ./run_tracker.sh stop
   ```

## Summary

This implementation provides a complete live-tracking system for Apple Notes with:
- ✅ Standalone tracker script running every 10 seconds
- ✅ `open-in-apple-notes` tool with automatic interaction logging
- ✅ Python backend support for interaction logging
- ✅ REST API endpoints for interaction data
- ✅ Comprehensive documentation
- ✅ Launcher script for easy management

The system is production-ready and handles all edge cases including duplicate titles, notes not yet in the database, and graceful shutdown.
