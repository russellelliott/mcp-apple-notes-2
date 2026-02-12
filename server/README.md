# MCP Apple Notes

![MCP Apple Notes](./images/logo.png)

A [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) server that enables fetching, embedding, and vector storage of your Apple Notes. This allows AI assistants like Claude to access your Apple Notes data efficiently.

![MCP Apple Notes](./images/demo.png)

## Features

- 🧠 **Device Embeddings** — Generates `bge-small-en-v1.5` embeddings on-device
- 📊 **Intelligent Chunking** — Automatic semantic chunking with 400-token chunks and 50-token overlap
- 🚀 **Vector Storage** — Using [LanceDB](https://lancedb.github.io/lancedb/) for fast vector operations
- 🤖 **MCP-Compatible** — Works with Claude Desktop and other MCP-aware assistants
- 🍎 **Native Integration** — Direct Apple Notes access via JXA
- 🏃‍♂️ **Fully Local** — No API keys needed, runs entirely on your machine

## Prerequisites

- [Bun](https://bun.sh/docs/installation)
- [Claude Desktop](https://claude.ai/download) (optional, for MCP integration)
- macOS (for Apple Notes access)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RafalWilinski/mcp-apple-notes
cd mcp-apple-notes
```

2. Install dependencies:

```bash
bun install
```

## Quick Start

### 1. Index Your Notes (Required First Step)

```bash
# Incremental update (Default) - Only processes new/modified notes
bun cli.ts

# Fresh rebuild (full reindex) - Resets DB and processes everything
bun cli.ts --mode=fresh
```

## Indexing Modes

The CLI supports different modes for indexing your notes. By default, it uses **incremental** mode to save time and resources.

### Incremental Modes

Both incremental strategies share these core features:
- **Smart Caching**: Tracks modification dates of notes to only process what has changed.
- **Efficient**: Skips notes that haven't been modified since the last run.
- **Fail-safe**: Keeps tracking of successfully processed notes even if the process is interrupted.

#### Default Incremental
`bun cli.ts` or `bun cli.ts --mode=incremental`

- **Usage**: Good for daily usage or when you want to control the batch size.
- **Specific**: Useful when you want to process a specific number of recent notes (via `--max`).

#### Incremental Since
`bun cli.ts --mode=incremental-since`

- **Date-Based**: Fetches all notes modified after the latest modification date in the database.
- **Fastest for Large Sets**: Avoids checking metadata for every single note if you haven't synced in a while.
- **Best for**: Processing all notes since the last run without knowing the exact count.

### Fresh Mode
`bun cli.ts --mode=fresh`

- **Complete Reset**: Wipes the existing database and cache.
- **Full Reindex**: Fetches and embeds every single note from scratch.
- **Verification**: Asks for confirmation before proceeding to prevent accidental data loss.
- **Best for**: First run, major updates to embedding model, or if database gets corrupted.

### Configuration

- `--max=<number>`: Limit the number of notes to process (useful for testing). Default: Unlimited.
- `--table=<name>`: Specify a custom LanceDB table name. Default: `notes`.

## Usage Methods

### Option 1: Claude Desktop Integration

1. Open Claude Desktop → Settings → Developer → Edit Config
2. Add entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-machine": {
      "command": "/Users/<YOUR_USER_NAME>/.bun/bin/bun",
      "args": ["/Users/<YOUR_USER_NAME>/path-to/mcp-apple-notes/index.ts"]
    }
  }
}
```

3. Restart Claude Desktop and ask: "Index my notes"

### Option 2: Command Line

Run scripts directly with Bun:

```bash
# Index notes (Incremental default)
bun cli.ts
```

## Project Structure

| File | Purpose |
|------|---------|
| `index.ts` | Core functions: embeddings |
| `cli.ts` | Note indexing and database management |
| `sync-db-cache.ts` | Diagnostic tool for DB/cache sync |

## Storage

- **Database**: `~/.mcp-apple-notes/data` (LanceDB vector store)
- **Cache**: `~/.mcp-apple-notes/notes-cache.json` (indexed notes backup)

## Dried Documentation

## Troubleshooting

**Check logs:**
```bash
tail -n 50 -f ~/Library/Logs/Claude/mcp-server-local-machine.log
# or
tail -n 50 -f ~/Library/Logs/Claude/mcp.log
```

**Index issues:**
- Ensure full disk access is enabled for your terminal
- Try `bun cli.ts --mode=fresh --max=100` to test with a small set first
- Check that Apple Notes.app is running

**Verify data:**
- Index first: `bun cli.ts --mode=fresh`
- Verify database exists: `ls ~/.mcp-apple-notes/data`
## Todos

- [ ] Apple notes returned as HTML → convert to Markdown
- [ ] Recursive text splitter for chunking
- [ ] Configurable embeddings model
- [ ] Advanced DB queries and purge functionality
- [x] Storing notes via Claude

**Verify data:**
- Index first: `bun cli.ts --mode=fresh`
- Verify database exists: `ls ~/.mcp-apple-notes/data`
- [LanceDB](https://lancedb.github.io/)
- [Hugging Face Embeddings](https://huggingface.co/sentence-transformers/bge-small-en-v1.5)
- Original PR with batching improvements: https://github.com/RafalWilinski/mcp-apple-notes/pull/3

```bash
=== Indexing Complete ===
📊 Stats:
• Total notes found: 14012
• Successfully indexed: 13983 notes
• Failed to process: 29 notes
• Time taken: 35550.16 seconds
```

```
35550.16 seconds ÷ 3600 seconds/hour ≈ 9.875 hours
Average rate = 13983 notes ÷ 35550.16 seconds ≈ 0.3934 notes/second
```

results from initial
- embedding model isn't very good
  - doing a search query yields irrelevant results
  - no normalization
  - old model (`Xenova/all-MiniLM-L6-v2`) is very small. using `Xenova/all-MiniLM-L6-v2` instead
- very poor text preprocessing
- markdown library (`turndown`) caused weird whitespace and formatting. iterating on this, i decided to convert it to plain text instead.
- to save time, starting off by indexing 1000 notes
- improved note batching and parellel processing

results from that change

```bash
=== Indexing Complete ===
📊 Stats:
• Total notes found: 1000
• Successfully indexed: 1000 notes
• Failed to process: 0 notes
• Time taken: 2176.45 seconds
```

```
2176.45 seconds ÷ 3600 seconds/hour ≈ 0.60457 hours (~36.27 minutes)
1000 notes ÷ 2176.45 seconds ≈ 0.4594 notes/second
```

results from that change

```bash
=== Indexing Complete ===
📊 Stats:
• Total notes found: 1000
• Successfully indexed: 1000 notes
• Failed to process: 0 notes
• Time taken: 2176.45 seconds
```

```
2176.45 seconds ÷ 3600 seconds/hour ≈ 0.60457 hours (~36.27 minutes)
1000 notes ÷ 2176.45 seconds ≈ 0.4594 notes/second
```

**Key improvements:**
- ✅ Upgraded to `Xenova/bge-small-en-v1.5` (better semantic understanding)
- ✅ Added embedding normalization (`normalize: true`)
- ✅ Replaced TurndownService with custom HTML-to-plaintext converter
- ✅ Implemented parallel processing (5 notes at once)
- ✅ Reduced delays and timeouts
- ✅ 16.8% faster processing rate

**Next:** Test search quality with new embedding model before scaling to full dataset.

still doesnt work

do 512 char substring for embeddings `.substring(0, 512)`
```
=== Indexing Complete ===
📊 Stats:
• Total notes found: 100
• Successfully indexed: 100 notes
• Failed to process: 0 notes
• Time taken: 222.96 seconds
```

still doesnt work

do 512 char substring for embeddings `.substring(0, 512)`
```
=== Indexing Complete ===
📊 Stats:
• Total notes found: 100
• Successfully indexed: 100 notes
• Failed to process: 0 notes
• Time taken: 222.96 seconds
```

```
222.96 seconds ÷ 60 ≈ 3.72 minutes
100 notes ÷ 222.96 seconds ≈ 0.4487 notes/second
```

**Latest improvements:**
- ✅ Added 512-character limit to `cleanText()` function
- ✅ Should improve embedding quality and consistency
- ✅ Reduced memory usage during embedding generation
- ✅ Performance rate consistent at ~0.45 notes/second

**Next:** Test if 512-char limit improved search relevance with simple queries.


new problem: how to deal with old notes?
current implementation: when you do index, old notes remain


added system to check if notes modified or not. skips over ones that are unchanged


The key improvements include:

Smart Table Creation: The createNotesTableSmart function supports both fresh rebuilds and incremental updates
Change Detection: Compares modification dates to only process changed notes
Efficient Processing: Skips unchanged notes, dramatically reducing processing time for large collections
Better CLI: Supports `--mode=fresh` or `--mode=incremental` and `--max=N` arguments
Detailed Stats: Shows exactly what was added, updated, or skipped

```
=== Indexing Complete ===
📊 Stats:
• Total processed: 100 notes
• New notes added: 1
• Notes updated: 12
• Notes skipped (unchanged): 87
• Failed: 0 notes
• Time taken: 222.63 seconds
```

```
=== Indexing Complete ===
📊 Stats:
• Total processed: 100 notes
• New notes added: 100
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 0 notes
• Time taken: 213.60 seconds
```


```
=== Indexing Complete ===
📊 Stats:
• Total processed: 215 notes
• New notes added: 100
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 0 notes
• Time taken: 222.36 seconds
```

=== Indexing Complete ===
📊 Stats:
• Notes processed: 100
• Chunks created: 197
• New notes added: 100
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 0 notes
• Time taken: 227.50 seconds

// ...existing code...

## Recent Improvements

### Chunking & Embedding Enhancements

**Better Embedding Model & Processing:**
- ✅ Upgraded from `all-MiniLM-L6-v2` to `bge-small-en-v1.5` for improved semantic understanding
- ✅ Added embedding normalization (`normalize: true`) for better similarity calculations
- ✅ Replaced TurndownService with custom HTML-to-plaintext converter that preserves formatting
- ✅ Enhanced text preprocessing with proper cleaning and tokenization

**Smart Chunking System:**
- ✅ Implemented intelligent text chunking with 400-token chunks and 50-token overlap
- ✅ Preserves document structure by splitting on natural boundaries (paragraphs, sentences)
- ✅ Handles edge cases with fallback chunking strategies
- ✅ Each note can generate multiple searchable chunks for better retrieval

**Incremental Indexing:**
- ✅ Smart update detection - only processes modified notes
- ✅ Compares modification dates to skip unchanged content
- ✅ Dramatically faster re-indexing (87% of notes skipped in typical runs)
- ✅ Fresh rebuild option available with `--mode=fresh`

**Performance Optimizations:**
- ✅ Parallel processing of notes (5 notes simultaneously)
- ✅ Optimized batching with progress tracking
- ✅ Reduced timeouts and delays for faster processing
- ✅ Memory-efficient chunk creation and storage

### CLI Improvements

**New Command Line Interface:**
```bash
# Fresh rebuild of entire database
bun run index-notes --mode=fresh

# Incremental updates (default)
bun run index-notes --mode=incremental

# Limit processing to specific number of notes
bun run index-notes --max=100

# Combine options
bun run index-notes --mode=fresh --max=500
```

**Better Progress Reporting:**
- Real-time batch processing updates
- Detailed statistics on new, updated, and skipped notes
- Performance metrics and timing information
- Error reporting with detailed failure logs

### Performance Results

**Before improvements:**
- ~0.39 notes/second (9.9 hours for 14k notes)
- Full re-processing on every run

**After improvements:**
- ~0.45 notes/second for new notes
- 87% skip rate for unchanged notes on incremental runs
- ~3-4 minutes to process 100 notes (including chunking)

### Common Issues

**Slow Initial Indexing:**
- First-time indexing is slower due to embedding generation
- Use `--max=100` to test with a subset of notes first
- Subsequent runs are much faster with incremental updates


=== Indexing Complete ===
📊 Stats:
• Notes processed: 100
• Chunks created: 197
• New notes added: 100
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 0 notes
• Time taken: 214.37 seconds




=== Indexing Complete ===
📊 Stats:
• Notes processed: 100
• Chunks created: 197
• New notes added: 100
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 0 notes
• Time taken: 209.88 seconds



=== Indexing Complete ===
📊 Stats:
• Notes processed: 14054
• Chunks created: 44394
• New notes added: 14054
• Notes updated: 0
• Notes skipped (unchanged): 0
• Failed: 29 notes
• Time taken: 30809.11 seconds



Chat with apple notes
https://github.com/yashgoenka/chat-apple-notes
- Limited by OpenAI quotas
- Does it work with larger apple notes?

might use as reference for how to search

Notes have chunk content as well as the entire content across all the notes
- could be used to find all the chunks of the notes?

Split notes into categories
- endpoint to get content of the note


Look at that one repo that has used ChatGPT embeddings and such; how does it retrieve relevant data to ask questions on. Use free models for that.

That repo; uses notes://showNote?identifier=id to open specific apple note?

uses OpenAI client; could look into using OpenRouter to use free models?

```python
EXTRACT_SCRIPT = """
tell application "Notes"
   repeat with eachNote in every note
      set noteId to the id of eachNote
      set noteTitle to the name of eachNote
      set noteBody to the body of eachNote
      set noteCreatedDate to the creation date of eachNote
      set noteCreated to (noteCreatedDate as «class isot» as string)
      set noteUpdatedDate to the modification date of eachNote
      set noteUpdated to (noteUpdatedDate as «class isot» as string)
      set noteContainer to container of eachNote
      set noteFolderId to the id of noteContainer
      log "{split}-id: " & noteId & "\n"
      log "{split}-created: " & noteCreated & "\n"
      log "{split}-updated: " & noteUpdated & "\n"
      log "{split}-folder: " & noteFolderId & "\n"
      log "{split}-title: " & noteTitle & "\n\n"
      log noteBody & "\n"
      log "{split}{split}" & "\n"
   end repeat
end tell
""".strip()
```

maybe for fetching note data, get the note id?

LMStudio; run LLMs locally on your machine
https://lmstudio.ai/

https://lmstudio.ai/blog/lmstudio-v0.3.17

https://github.com/RafalWilinski/mcp-apple-notes
Open the claude_desktop_config.json and add the following entry:
```json
{
  "mcpServers": {
    "local-machine": {
      "command": "/Users/<YOUR_USER_NAME>/.bun/bin/bun",
      "args": ["/Users/<YOUR_USER_NAME>/apple-notes-mcp/index.ts"]
    }
  }
}
```
advantage of LMStudio is that it's free to integrate MCP servers, while Claude requires paid subscription for that, which is something the repos failed to mention.
https://www.anthropic.com/pricing

you may notice that some notes timeout sometimes and retry with extended timeout. this is typical and works out in the end.


Get url of apple notes
https://discourse.hookproductivity.com/t/using-the-built-in-notes-url-scheme/6071
- It needs full disk access
- Chat with apple notes
- https://github.com/yashgoenka/chat-apple-notes

`test-note-id.ts` shows how to get the ID of a given note.

it might be something to look into? would need full disk access. the method with just doing it by title works without full disk access.
- honestly seems like alot of effort. the solution i have right now is sufficient
- if i really wanted to, i could somehow track the number of notes with a given title, then add those one by one?

This approach is better, especially when more than one note has the same title.

https://claude.ai/chat/815696fc-ca9e-4bc5-a3d5-1e3c71fe9f57