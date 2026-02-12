# MCP Server

![MCP Apple Notes](./images/logo.png)

A [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) server that enables fetching, embedding, and vector storage of your Apple Notes. This allows AI assistants like Claude to access your Apple Notes data efficiently.

> Originally based on [mcp-apple-notes](https://github.com/russellelliott/mcp-apple-notes), serving as the foundation for this server implementation.

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

1. Install dependencies:

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
    "apple-notes": {
      "command": "/Users/<YOUR_USER_NAME>/.bun/bin/bun",
      "args": ["/Users/<YOUR_USER_NAME>/path-to/mcp-apple-notes/server/index.ts"]
    }
  }
}
```

> **Note**: Make sure to point `args` to the absolute path of `server/index.ts` within this repository.

3. Restart Claude Desktop and ask: "Index my notes" or "What's in my notes about [Topic]?"

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
