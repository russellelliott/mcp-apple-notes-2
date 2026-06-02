# Apple Notes Live Tracker

A standalone background process that tracks note interactions (opens and modifications) in a separate LanceDB table called `notes_interactions`.

## Features

- **Automatic Tracking**: Runs every 10 seconds to poll open note windows
- **Persistent Storage**: Stores interaction history in a separate LanceDB table
- **Loose Relationship**: Tracks notes even if they don't exist in the main `notes` table yet
- **Event Logging**: Records "opened" and "modified" events with timestamps

## Schema

The `notes_interactions` table has the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `title` | String | Note title (primary key) |
| `last_opened` | String | ISO timestamp of most recent interaction |
| `interaction_log` | String | JSON array of events: `[{"dt": "<timestamp>", "type": "opened" \| "modified"}]` |

## Usage

### Running the Tracker

```bash
# Make executable
chmod +x tracker.js

# Run the tracker
node tracker.js
```

The tracker will:
1. Create the `notes_interactions` table if it doesn't exist
2. Poll Apple Notes every 10 seconds for open windows
3. Log "opened" events for each open note
4. Continue running until you press Ctrl+C

### Stopping the Tracker

Press `Ctrl+C` in the terminal to gracefully stop the tracker.

## Integration with Main Application

### Opening Notes

Use the `open-in-apple-notes` tool to open a specific note:

```typescript
// Opens a note and automatically logs the interaction
await callTool({
  name: "open-in-apple-notes",
  arguments: {
    title: "My Note Title",
    creation_date: "2024-01-15 10:30:00" // Optional for duplicate titles
  }
});
```

### Fetching Interaction Data

The backend server provides endpoints to fetch interaction data:

#### Get Last Opened Timestamp

```bash
curl "http://localhost:8000/interaction/My%20Note"
```

Response:
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

#### List All Interactions

```bash
curl "http://localhost:8000/interactions/list?limit=50"
```

Response:
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

### Logging Modifications

When the main sync script detects a note modification, it should log the event:

```python
from backend.scripts.main import NotesDatabase

db = NotesDatabase()
db.log_note_modified("My Note Title")
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

## Architecture

```
┌──────────────────────┐
│  tracker.js          │
│  (runs every 10s)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ notes_interactions   │
│ (separate table)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  server.py           │
│  (API endpoints)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Frontend UI         │
│  (display last_opened)│
└──────────────────────┘
```

## Troubleshooting

### Tracker Not Starting

1. Check that Node.js is installed: `node --version`
2. Check that the `run-jxa` package is installed
3. Check that Apple Notes has automation permissions in System Preferences

### Notes Not Being Tracked

1. Ensure Apple Notes is running
2. Check that Notes windows are actually open
3. Verify the `notes_interactions` table exists: `ls ~/.mcp-apple-notes/data/`

### Interaction Log Not Updating

1. Check the tracker logs for errors
2. Verify the tracker is still running
3. Check disk space availability

## Future Enhancements

- [ ] Add "focused" event type for when a note window gains focus
- [ ] Add duration tracking (how long a note was open)
- [ ] Add analytics endpoint for interaction patterns
- [ ] Support for note creation/deletion events
