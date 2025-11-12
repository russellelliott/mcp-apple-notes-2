#!/usr/bin/env python3
"""Fetch notes, convert to plain text, chunk them and write chunk data files.

This script avoids importing the full `main.py` (and heavy ML libs) and uses AppleScript
via `osascript` to fetch note metadata and content. It then performs character-based
chunking using the same heuristics as the main code.

Usage:
  python scripts/fetch_and_chunk_notes.py --limit 100

Logs progress to stdout; writes chunk JSONL files into ./data/chunks_batch_{i}.jsonl
"""
import argparse
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import re
import sys
import threading
import queue
import hashlib
import uuid
# Ensure the repository root is on sys.path so `import scripts.*` works when this
# script is executed directly (e.g. `python scripts/fetch_and_chunk_notes.py`).
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# import helper functions from scripts so we don't reimplement AppleScript here
from scripts.print_note_titles import fetch_titles as helper_fetch_titles
from scripts.get_note_content import fetch_note_content as helper_fetch_note_content


# Configuration (token-based defaults mapped to approximate char counts)
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
MAX_CHUNK_SIZE_TOKENS = 512
AVG_CHARS_PER_TOKEN = 4  # heuristic

CHUNK_MAX_CHARS = min(CHUNK_SIZE_TOKENS * AVG_CHARS_PER_TOKEN, MAX_CHUNK_SIZE_TOKENS * AVG_CHARS_PER_TOKEN)
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * AVG_CHARS_PER_TOKEN

FETCH_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE = 10
DB_BATCH_SIZE = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Persistence configuration (use LanceDB in the user's home directory)
DATA_DIR = Path.home() / ".mcp-apple-notes-2"
DB_PATH = DATA_DIR / "data"
CACHE_PATH = DATA_DIR / "last-sync.txt"
# LanceDB table name (configurable)
TABLE_NAME = "notes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import lancedb and pyarrow for storage
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import os
import shutil
import time

# Load embedding model eagerly (no lazy loading)
MODEL = SentenceTransformer(MODEL_NAME)


# ============================================================================
# Cache Management
# ============================================================================

class NotesCache:
    """Manage incremental indexing cache using last-run timestamp."""
    
    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
    
    def load_last_sync_time(self) -> float:
        """Load the timestamp of last successful sync. Returns 0 if no cache."""
        if not self.cache_path.exists():
            return 0
        try:
            with open(self.cache_path, 'r') as f:
                content = f.read().strip()
                # Extract just the first line (timestamp)
                return float(content.split('\n')[0])
        except (ValueError, OSError):
            return 0
    
    def save_last_sync_time(self) -> None:
        """Save current time as last sync timestamp with human-readable ISO format."""
        from datetime import datetime
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        current_time = time.time()
        iso_time = datetime.fromtimestamp(current_time).isoformat()
        with open(self.cache_path, 'w') as f:
            f.write(f"{current_time}\n{iso_time}\n")
    
    def filter_by_modification_time(self, notes: List[Dict[str, str]]) -> tuple:
        """Filter notes into (changed, unchanged) based on modification_date vs last sync.
        
        Changed = modification_date is newer than last sync time.
        Unchanged = modification_date is older than or equal to last sync time.
        """
        last_sync = self.load_last_sync_time()
        if last_sync == 0:
            # No cache yet; all notes are "new"
            return notes, []
        
        changed = []
        unchanged = []
        
        for note in notes:
            mod_date_str = note.get('modification_date', '')
            if not mod_date_str:
                # If no modification date, treat as changed
                changed.append(note)
                continue
            
            # Parse ISO datetime and convert to timestamp
            try:
                from datetime import datetime
                mod_dt = datetime.fromisoformat(mod_date_str.replace('Z', '+00:00'))
                mod_timestamp = mod_dt.timestamp()
            except (ValueError, AttributeError):
                # Parse error; treat as changed
                changed.append(note)
                continue
            
            if mod_timestamp > last_sync:
                changed.append(note)
            else:
                unchanged.append(note)
        
        return changed, unchanged


def fetch_titles(limit: int) -> List[Dict[str, str]]:
    """Delegate to helper function in `scripts/print_note_titles.py`."""
    return helper_fetch_titles(limit)


def fetch_note_content(item: Dict[str, str]) -> Dict[str, str]:
    """Delegate to helper function in `scripts/get_note_content.py`.

    The helper returns a dict with keys 'title','creation_date','modification_date','content'.
    """
    title = item.get('title', '')
    creation = item.get('creation_date', '')
    return helper_fetch_note_content(title, creation)


# We intentionally do NOT reimplement HTML sanitization here; the helper script
# returns raw content (or a JSON-wrapped raw). The pipeline below will use the
# raw `content` field as-is and combine it with the title.


def create_chunks(text: str, max_chars: int = int(CHUNK_MAX_CHARS), overlap_chars: int = int(CHUNK_OVERLAP_CHARS)) -> List[str]:
    if not text:
        return ['']
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start + max_chars * 0.7:
                end = paragraph_break
            else:
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break > start + max_chars * 0.7:
                    end = sentence_break + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end - overlap_chars)
    return chunks if chunks else ['']


def _lancetable_to_arrow(tbl):
    """Convert a LanceDB table to PyArrow table, trying multiple API methods."""
    for meth in ("to_arrow_table", "to_table", "to_pyarrow", "to_arrow"):
        if hasattr(tbl, meth):
            return getattr(tbl, meth)()
    # Fallback to pandas
    if hasattr(tbl, "to_pandas"):
        return pa.Table.from_pandas(tbl.to_pandas())
    raise RuntimeError("Cannot convert LanceTable to pyarrow; please update lancedb")


def write_chunks_to_lancedb(chunks: list[dict], batch_index: int):
    """Write a batch of chunks to LanceDB, handling schema mismatches by rebuilding the table."""
    if not chunks:
        return

    # Connect to LanceDB
    db = lancedb.connect(str(DB_PATH))

    # Convert chunks to PyArrow table
    pa_table = pa.Table.from_pylist(chunks)

    # Try to open existing table; if it doesn't exist, create it fresh
    try:
        tbl = db.open_table(TABLE_NAME)
    except Exception:
        # Table doesn't exist; create it fresh
        try:
            db.create_table(TABLE_NAME, pa_table)
            return
        except Exception:
            # Race condition: table was created elsewhere
            tbl = db.open_table(TABLE_NAME)

    # Table exists; try to append
    try:
        if hasattr(tbl, "append"):
            tbl.append(pa_table)
        elif hasattr(tbl, "add"):
            tbl.add(pa_table)
        elif hasattr(tbl, "insert"):
            tbl.insert(pa_table)
        else:
            raise RuntimeError("Table has no append/add/insert method")
    except Exception as e:
        # Schema mismatch or other error. Check if schemas are truly incompatible.
        try:
            existing = _lancetable_to_arrow(tbl)
            existing_schema = set(existing.schema.names)
            new_schema = set(pa_table.schema.names)
            
            # If new schema has fields the old doesn't (e.g., 'embedding'), we need to rebuild
            if new_schema != existing_schema:
                print(f"🔄 Schema mismatch detected. Existing: {existing_schema}, New: {new_schema}")
                print(f"   Deleting and rebuilding table with new schema...")
                
                # Delete the old DB and create fresh with new schema
                shutil.rmtree(str(DB_PATH))
                DB_PATH.mkdir(parents=True, exist_ok=True)
                db_fresh = lancedb.connect(str(DB_PATH))
                db_fresh.create_table(TABLE_NAME, pa_table)
                return
            
            # Same schema but different types; do field-by-field alignment
            for name in (new_schema - existing_schema):
                field = pa_table.schema.field(name)
                nulls = pa.nulls(len(existing), type=field.type)
                existing = existing.append_column(name, nulls)

            for name in (existing_schema - new_schema):
                field = existing.schema.field(name)
                nulls = pa.nulls(len(pa_table), type=field.type)
                pa_table = pa_table.append_column(name, nulls)

            # Reorder to existing column order
            order = list(existing.schema.names)
            pa_table = pa_table.select(order)

            # Cast types to match
            pa_table = pa_table.cast(existing.schema)
            combined = pa.concat_tables([existing, pa_table])

            # Rebuild table atomically
            tmp_dir = str(DB_PATH) + f".tmp-{int(time.time())}"
            os.makedirs(tmp_dir, exist_ok=True)
            db_tmp = lancedb.connect(tmp_dir)
            db_tmp.create_table(TABLE_NAME, combined, schema=existing.schema)

            # Atomic replace
            shutil.rmtree(str(DB_PATH))
            os.replace(tmp_dir, str(DB_PATH))
            return
        except Exception as merge_err:
            print(f"❌ Failed to handle schema mismatch: {merge_err}")
            raise


def _processing_worker(processing_q: "queue.Queue", embedding_q: "queue.Queue", worker_id: int):
    while True:
        item = processing_q.get()
        if item is None:
            # propagate sentinel and exit
            processing_q.task_done()
            break
        idx, note = item
        title = note.get('title', '<untitled>')
        raw = note.get('content') or note.get('raw') or ''
        full_text = f"{title}\n\n{raw}"
        ct_chunks = create_chunks(full_text)
        total_chunks = len(ct_chunks)
        chunks = []
        creation = note.get('creation_date', '')
        modification = note.get('modification_date', '')
        for ci, chunk_text in enumerate(ct_chunks):
            # Generate unique ID for each chunk
            chunk_id = str(uuid.uuid4())
            chunk_obj = {
                'id': chunk_id,
                'title': title,
                'content': raw,
                'creation_date': creation,
                'modification_date': modification,
                'chunk_index': ci,
                'total_chunks': total_chunks,
                'chunk_content': chunk_text,
            }
            chunks.append(chunk_obj)
        # push all chunks for embedding
        for c in chunks:
            embedding_q.put(c)
        processing_q.task_done()


def _embedding_worker(embedding_q: "queue.Queue", writer_q: "queue.Queue"):
    model = MODEL
    buffer = []
    while True:
        item = embedding_q.get()
        if item is None:
            # flush buffer
            if buffer:
                texts = [c['chunk_content'] for c in buffer]
                embs = model.encode(texts, convert_to_numpy=True)
                for c, e in zip(buffer, embs.tolist() if hasattr(embs, 'tolist') else embs):
                    c['embedding'] = e.tolist() if hasattr(e, 'tolist') else list(e)
                    writer_q.put(c)
                buffer = []
            # propagate sentinel to writer and exit
            writer_q.put(None)
            embedding_q.task_done()
            break

        buffer.append(item)
        # batch embeddings
        if len(buffer) >= EMBEDDING_BATCH_SIZE:
            texts = [c['chunk_content'] for c in buffer]
            embs = model.encode(texts, convert_to_numpy=True)
            for c, e in zip(buffer, embs.tolist() if hasattr(embs, 'tolist') else embs):
                c['embedding'] = e.tolist() if hasattr(e, 'tolist') else list(e)
                writer_q.put(c)
            buffer = []
        embedding_q.task_done()


def _writer_worker(writer_q: "queue.Queue", batch_index: int):
    buffer = []
    while True:
        item = writer_q.get()
        if item is None:
            # flush remaining buffer
            if buffer:
                write_chunks_to_lancedb(buffer, batch_index)
                buffer = []
            writer_q.task_done()
            break
        buffer.append(item)
        if len(buffer) >= DB_BATCH_SIZE:
            write_chunks_to_lancedb(buffer, batch_index)
            buffer = []
        writer_q.task_done()



def process(limit: int, force: bool = False):
    titles = fetch_titles(limit)
    total = len(titles)
    print(f"📋 Found {total} note titles")
    if total == 0:
        return

    # Filter by cache to skip unchanged notes (unless --force is used)
    cache = NotesCache(CACHE_PATH)
    
    if force:
        print(f"🔄 Force mode: Processing all {total} notes (ignoring cache)")
        changed = titles
        unchanged = []
    else:
        changed, unchanged = cache.filter_by_modification_time(titles)
        print(f"📊 Cache check: {len(changed)} changed, {len(unchanged)} unchanged")
        if unchanged:
            print(f"   ⏭️  Skipping {len(unchanged)} unchanged notes")
        titles = changed
        if not titles:
            print("✅ All notes up-to-date!")
            cache.save_last_sync_time()
            return
    
    total = len(changed)

    overall_processed = 0
    batch_count = (total + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE

    for bidx in range(batch_count):
        i0 = bidx * FETCH_BATCH_SIZE
        batch = titles[i0:i0 + FETCH_BATCH_SIZE]
        print(f"\n📦 Processing batch {bidx+1}/{batch_count} ({len(batch)} notes):")

        # Pipeline: producer (fetch) -> processing workers (chunk) -> embedding worker -> writer
        processing_q: "queue.Queue" = queue.Queue()
        embedding_q: "queue.Queue" = queue.Queue()
        writer_q: "queue.Queue" = queue.Queue()

        num_proc_workers = min(8, max(1, len(batch)))
        proc_threads = []
        for wid in range(num_proc_workers):
            t = threading.Thread(target=_processing_worker, args=(processing_q, embedding_q, wid), daemon=True)
            t.start()
            proc_threads.append(t)

        emb_thread = threading.Thread(target=_embedding_worker, args=(embedding_q, writer_q), daemon=True)
        emb_thread.start()

        writer_thread = threading.Thread(target=_writer_worker, args=(writer_q, bidx+1), daemon=True)
        writer_thread.start()

        success = 0
        # Producer: submit fetch jobs and enqueue results to processing_q
        with ThreadPoolExecutor(max_workers=FETCH_BATCH_SIZE) as ex:
            futures = {}
            for idx, it in enumerate(batch, 1):
                title_preview = it.get('title', '<untitled>')
                print(f"  📄 [{idx}/{len(batch)}] Starting fetch: \"{title_preview}\"", flush=True)
                futures[ex.submit(fetch_note_content, it)] = (idx, it)

            completed_count = 0
            for fut in as_completed(list(futures.keys())):
                idx, item = futures.pop(fut)
                completed_count += 1
                try:
                    note = fut.result()
                except Exception as e:
                    note = {'error': str(e), 'title': item.get('title')}

                title = note.get('title', '<untitled>')
                print(f"  📄 [{completed_count}/{len(batch)}] Finished fetch: \"{title}\"", flush=True)

                if note.get('error'):
                    print(f"  ❌ [{completed_count}] Failed: \"{title}\" - {note.get('error')}", flush=True)
                    continue

                print(f"    📅 Created: {note.get('creation_date','')}  ✏️ Modified: {note.get('modification_date','')}", flush=True)
                print(f"    ➕ Starting to add note: \"{title}\"", flush=True)

                # enqueue for processing workers
                processing_q.put((completed_count, note))
                success += 1

        # signal processing workers to stop
        for _ in range(num_proc_workers):
            processing_q.put(None)

        # wait for processing workers to finish
        processing_q.join()

        # signal embedding worker to stop (single worker)
        embedding_q.put(None)

        # wait for embedding and writer to finish
        embedding_q.join()
        writer_q.join()

        # Make sure worker threads finish
        for t in proc_threads:
            t.join(timeout=1)
        emb_thread.join(timeout=1)
        writer_thread.join(timeout=1)

        overall_processed += success
        print(f"📊 Batch {bidx+1} complete: {success}/{len(batch)} successful")
        print(f"🎯 Overall progress: {overall_processed}/{total} notes processed")

    # Always save cache timestamp for future runs
    cache.save_last_sync_time()
    print(f"💾 Saved sync timestamp to cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Limit number of notes to fetch (0 = all)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing all notes, ignore cache, then update cache timestamp')
    args = parser.parse_args()
    start = time.time()
    process(args.limit if args.limit > 0 else 0, force=args.force)
    print(f"\nCompleted in {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()
