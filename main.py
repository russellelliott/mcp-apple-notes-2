#!/usr/bin/env python3
"""
Apple Notes MCP Server - Python Edition
Semantic search and clustering for Apple Notes using LanceDB + GPU embeddings
"""

import os
import json
import asyncio
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator
from dataclasses import dataclass, asdict
from datetime import datetime

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Vector DB and ML
import lancedb
import pyarrow as pa
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path.home() / ".mcp-apple-notes-2"
DB_PATH = DATA_DIR / "data"
CACHE_PATH = DATA_DIR / "notes-cache.json"

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 400  # tokens
CHUNK_OVERLAP = 50  # tokens

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Using NVIDIA GPU (CUDA)")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU (no GPU acceleration)")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class NoteMetadata:
    """Metadata for a note"""
    title: str
    creation_date: str
    modification_date: str
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class Note:
    """Full note with content"""
    title: str
    content: str
    creation_date: str
    modification_date: str
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class NoteChunk:
    """A chunk of a note with embedding"""
    title: str
    content: str  # Full note content
    creation_date: str
    modification_date: str
    chunk_index: int
    total_chunks: int
    chunk_content: str
    # Clustering fields (optional)
    cluster_id: str = ""
    cluster_label: str = ""
    cluster_confidence: str = ""
    cluster_summary: str = ""
    last_clustered: str = ""


# ============================================================================
# Apple Notes Extraction
# ============================================================================

class AppleNotesExtractor:
    """Extract notes from Apple Notes using AppleScript"""
    
    EXTRACT_METADATA_SCRIPT = """
    tell application "Notes"
        set noteList to {}
        repeat with eachNote in every note
            set noteTitle to the name of eachNote
            set noteCreatedDate to the creation date of eachNote
            set noteCreated to (noteCreatedDate as «class isot» as string)
            set noteUpdatedDate to the modification date of eachNote
            set noteUpdated to (noteUpdatedDate as «class isot» as string)
            
            set end of noteList to noteTitle & "|||" & noteCreated & "|||" & noteUpdated
        end repeat
        return noteList
    end tell
    """
    # NOTE: the old AppleScript returned a delimiter-separated string which caused issues
    # with content containing the delimiter. We'll use JavaScript for Automation (JXA)
    # via `osascript -l JavaScript` to return proper JSON strings instead.
    
    @staticmethod
    def run_applescript(script: str, language: str = "applescript") -> str:
        """Run a script (AppleScript or JXA) and return output"""
        cmd = ["osascript"]
        if language == "javascript":
            cmd += ["-l", "JavaScript", "-e", script]
        else:
            cmd += ["-e", script]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    
    @classmethod
    def get_note_by_title_and_date(cls, title: str, creation_date: str) -> Optional[Note]:
        """Get a specific note by title and creation date"""
        # Use AppleScript to fetch the note content and metadata.
        # We'll first search for the note by title (and optionally creation_date),
        # then print creation and modification dates followed by a delimiter and the note body.
        safe_title = title.replace('"', '\\"')
        safe_creation = creation_date or ""

        # AppleScript: find the matching note and print metadata then content separated by a marker
        script_meta = f"""
        set found to false
        tell application "Notes"
            set noteList to every note
            repeat with n in noteList
                try
                    set nTitle to name of n
                    set nCreation to (creation date of n) as «class isot» as string
                    if nTitle is "{safe_title}" then
                        if "{safe_creation}" is not "" and nCreation is not "{safe_creation}" then
                            -- skip
                        else
                            set c to nCreation
                            set m to ((modification date of n) as «class isot» as string)
                            -- print creation and modification on one line separated by tab
                            return c & "\t" & m
                        end if
                    end if
                end try
            end repeat
        end tell
        return ""
        """

        try:
            meta_out = cls.run_applescript(script_meta, language="applescript")
            if not meta_out:
                return None

            parts = meta_out.split('\t')
            creation = parts[0] if len(parts) > 0 else ""
            modification = parts[1] if len(parts) > 1 else ""

            # Now fetch the body for the same note
            script_body = f"""
            tell application "Notes"
                set noteList to every note
                repeat with n in noteList
                    try
                        if name of n is "{safe_title}" then
                            set nCreation to (creation date of n) as «class isot» as string
                            if "{safe_creation}" is not "" and nCreation is not "{safe_creation}" then
                                -- skip
                            else
                                return (body of n)
                            end if
                        end if
                    end try
                end repeat
            end tell
            return ""
            """

            body_out = cls.run_applescript(script_body, language="applescript")

            return Note(
                title=title,
                creation_date=creation,
                modification_date=modification,
                content=body_out or ""
            )
        except subprocess.CalledProcessError:
            return None
    
    @classmethod
    def get_all_notes_metadata(cls, limit: Optional[int] = None) -> List[NoteMetadata]:
        """Get metadata for all notes (fast).

        Uses JXA to return a JSON array of metadata objects. If `limit` is set,
        only the first `limit` notes (in the Notes app order) are returned.
        """
        print("📋 Fetching note metadata from Apple Notes (JSON)...")

        # Use AppleScript to fetch note metadata as tab-separated lines.
        apple_limit = limit if limit is not None else 0

        script = f"""
        on replaceText(theText, searchString, replacementString)
            set AppleScript's text item delimiters to searchString
            set theItems to every text item of theText
            set AppleScript's text item delimiters to replacementString
            set theText to theItems as string
            set AppleScript's text item delimiters to ""
            return theText
        end replaceText

        set outLines to ""
        tell application "Notes"
            set noteList to every note
            set totalCount to (count of noteList)
            set maxNotes to {apple_limit}
            if maxNotes = 0 then
                set limitCount to totalCount
            else
                if maxNotes < totalCount then
                    set limitCount to maxNotes
                else
                    set limitCount to totalCount
                end if
            end if

            repeat with i from 1 to limitCount
                try
                    set n to item i of noteList
                    set t to name of n
                    set c to (creation date of n) as «class isot» as string
                    set m to (modification date of n) as «class isot» as string

                    -- sanitize title (remove tabs/newlines)
                    set t1 to my replaceText(t, tab, " ")
                    set t2 to my replaceText(t1, return, " ")
                    set t3 to my replaceText(t2, linefeed, " ")

                    set outLines to outLines & t3 & tab & c & tab & m & linefeed
                on error errMsg
                    -- skip problematic note
                end try
            end repeat
        end tell

        return outLines
        """

        try:
            result = cls.run_applescript(script, language="applescript")
            if not result:
                print("📋 AppleScript returned no output")
                return []

            notes = []
            for line in result.splitlines():
                parts = line.split('\t')
                if not parts:
                    continue
                title = parts[0]
                creation = parts[1] if len(parts) > 1 else ""
                modification = parts[2] if len(parts) > 2 else ""
                notes.append(NoteMetadata(
                    title=title,
                    creation_date=creation,
                    modification_date=modification
                ))

            print(f"✅ Found {len(notes)} notes")
            return notes

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to extract notes: {e}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
            return []


# ============================================================================
# Text Processing
# ============================================================================

class TextProcessor:
    """Process and chunk text content"""
    
    @staticmethod
    def html_to_plain_text(html: str) -> str:
        """Convert HTML to plain text"""
        if not html:
            return ""
        
        # Use BeautifulSoup for robust HTML parsing
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean up whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    @staticmethod
    def create_chunks(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        Simple character-based approach (can be enhanced with token-aware splitting)
        """
        if not text or len(text) <= max_chars:
            return [text] if text else [""]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break on paragraph
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start + max_chars * 0.7:
                    end = paragraph_break
                else:
                    # Look for sentence break
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
            
            # Move start position with overlap
            start = max(start + 1, end - overlap_chars)
        
        return chunks if chunks else [""]


# ============================================================================
# Embeddings
# ============================================================================

class EmbeddingModel:
    """GPU-accelerated embedding generation"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Model loaded on {device}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Important for cosine similarity
            convert_to_numpy=True
        )
        return embeddings


# ============================================================================
# Cache Management
# ============================================================================

class NotesCache:
    """Manage notes cache for incremental updates"""
    
    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load cache from disk"""
        if not self.cache_path.exists():
            return None
        
        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Cache file corrupted, ignoring")
            return None
    
    def save(self, notes: List[NoteMetadata]) -> None:
        """Save notes metadata to cache"""
        cache = {
            "last_sync": datetime.now().isoformat(),
            "notes": [note.to_dict() for note in notes]
        }
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"💾 Saved {len(notes)} notes to cache")
    
    def identify_changes(
        self,
        current_notes: List[NoteMetadata],
        cached_notes: List[NoteMetadata]
    ) -> Dict[str, List[NoteMetadata]]:
        """Identify new, modified, and unchanged notes"""
        cached_map = {
            note.title: note for note in cached_notes
        }
        
        new_notes = []
        modified_notes = []
        unchanged_notes = []
        
        for note in current_notes:
            cached = cached_map.get(note.title)
            
            if not cached:
                new_notes.append(note)
            elif cached.modification_date != note.modification_date:
                modified_notes.append(note)
            else:
                unchanged_notes.append(note)
        
        return {
            "new": new_notes,
            "modified": modified_notes,
            "unchanged": unchanged_notes
        }


# ============================================================================
# LanceDB Integration
# ============================================================================

class NotesDatabase:
    """LanceDB vector database for notes"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.table = None
    
    def create_schema(self) -> pa.Schema:
        """Create PyArrow schema for notes table"""
        return pa.schema([
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("creation_date", pa.string()),
            pa.field("modification_date", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("total_chunks", pa.int32()),
            pa.field("chunk_content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            # Clustering fields
            pa.field("cluster_id", pa.string()),
            pa.field("cluster_label", pa.string()),
            pa.field("cluster_confidence", pa.string()),
            pa.field("cluster_summary", pa.string()),
            pa.field("last_clustered", pa.string()),
        ])
    
    def get_or_create_table(self, fresh: bool = False) -> lancedb.table.Table:
        """Get existing table or create new one"""
        table_name = "notes"
        
        if fresh and table_name in self.db.table_names():
            self.db.drop_table(table_name)
            print(f"🗑️ Dropped existing '{table_name}' table")
        
        if table_name not in self.db.table_names():
            # Create empty table with schema
            self.table = self.db.create_table(
                table_name,
                schema=self.create_schema()
            )
            print(f"✅ Created new '{table_name}' table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"📂 Opened existing '{table_name}' table")
        
        return self.table
    
    def add_chunks(self, chunks_data: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add chunks to the database in batches"""
        total = len(chunks_data)
        print(f"💾 Adding {total} chunks to database...")
        
        for i in range(0, total, batch_size):
            batch = chunks_data[i:i + batch_size]
            self.table.add(batch)
            print(f"  ✓ Added batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
        row_count = self.table.count_rows()
        print(f"✅ Database now has {row_count} chunks")
    
    def delete_note_chunks(self, title: str) -> None:
        """Delete all chunks for a specific note"""
        try:
            self.table.delete(f"title = '{title.replace(chr(39), chr(39)+chr(39))}'")
            print(f"  🗑️ Deleted chunks for '{title}'")
        except Exception as e:
            print(f"  ⚠️ Could not delete chunks for '{title}': {e}")
    
    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """Vector search for similar chunks"""
        results = self.table.search(query_vector).limit(limit).to_list()
        return results


# ============================================================================
# Indexing Pipeline
# ============================================================================

class NotesIndexer:
    """Main indexing pipeline"""
    
    def __init__(self):
        self.extractor = AppleNotesExtractor()
        self.processor = TextProcessor()
        self.embedder = EmbeddingModel()
        self.cache = NotesCache()
        self.db = NotesDatabase()
    
    async def index_notes(self, mode: str = "incremental", limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Index notes into vector database
        
        Args:
            mode: 'fresh' (reindex all) or 'incremental' (only new/modified)
        """
        start_time = datetime.now()
        print(f"\n🚀 Starting notes indexing (mode: {mode})")
        
        # Step 1: Get current notes metadata (optionally limited)
        current_notes = self.extractor.get_all_notes_metadata(limit=limit)
        
        if not current_notes:
            return {"error": "No notes found"}
        
        # Step 2: Determine which notes to process
        if mode == "fresh":
            notes_to_process = current_notes
            self.db.get_or_create_table(fresh=True)
        else:
            cached = self.cache.load()
            if cached:
                cached_notes = [NoteMetadata(**n) for n in cached["notes"]]
                changes = self.cache.identify_changes(current_notes, cached_notes)
                
                notes_to_process = changes["new"] + changes["modified"]
                skipped = len(changes["unchanged"])
                
                print(f"📊 Change analysis:")
                print(f"  • New: {len(changes['new'])}")
                print(f"  • Modified: {len(changes['modified'])}")
                print(f"  • Unchanged: {skipped}")
                
                if not notes_to_process:
                    print("✨ No changes detected!")
                    self.cache.save(current_notes)
                    return {
                        "processed": 0,
                        "chunks": 0,
                        "skipped": skipped,
                        "time_seconds": 0
                    }
                
                # Delete old chunks for modified notes
                self.db.get_or_create_table(fresh=False)
                for note_meta in changes["modified"]:
                    self.db.delete_note_chunks(note_meta.title)
            else:
                print("📁 No cache found, processing all notes")
                notes_to_process = current_notes
                self.db.get_or_create_table(fresh=False)
        
        # Step 3: Fetch full note content
        print(f"\n📥 Fetching content for {len(notes_to_process)} notes...")
        notes_with_content = []
        
        for i, note_meta in enumerate(notes_to_process, 1):
            note = self.extractor.get_note_by_title_and_date(
                note_meta.title,
                note_meta.creation_date
            )
            if note:
                notes_with_content.append(note)
                if i % 10 == 0:
                    print(f"  📄 Fetched {i}/{len(notes_to_process)}")
        
        print(f"✅ Fetched {len(notes_with_content)} notes")
        
        # Step 4: Process into chunks
        print(f"\n✂️ Creating chunks...")
        all_chunks = []
        
        for note in notes_with_content:
            plain_text = self.processor.html_to_plain_text(note.content)
            full_text = f"{note.title}\n\n{plain_text}"
            chunk_texts = self.processor.create_chunks(full_text)
            
            for idx, chunk_text in enumerate(chunk_texts):
                all_chunks.append(NoteChunk(
                    title=note.title,
                    content=plain_text,
                    creation_date=note.creation_date,
                    modification_date=note.modification_date,
                    chunk_index=idx,
                    total_chunks=len(chunk_texts),
                    chunk_content=chunk_text
                ))
        
        print(f"✅ Created {len(all_chunks)} chunks")
        
        # Step 5: Generate embeddings
        print(f"\n🧮 Generating embeddings on {DEVICE}...")
        chunk_texts = [chunk.chunk_content for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(chunk_texts, show_progress=True)
        
        # Step 6: Prepare data for database
        chunks_data = []
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk_dict = asdict(chunk)
            chunk_dict["vector"] = embedding.tolist()
            chunks_data.append(chunk_dict)
        
        # Step 7: Add to database
        self.db.add_chunks(chunks_data)
        
        # Step 8: Update cache
        self.cache.save(current_notes)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {
            "processed": len(notes_with_content),
            "chunks": len(all_chunks),
            "skipped": len(current_notes) - len(notes_to_process) if mode == "incremental" else 0,
            "time_seconds": elapsed
        }
        
        print(f"\n✨ Indexing complete in {elapsed:.1f}s!")
        return result


# ============================================================================
# Search
# ============================================================================

class NotesSearcher:
    """Semantic search over notes"""
    
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.db = NotesDatabase()
        self.db.get_or_create_table()
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search for notes
        
        Returns list of notes with relevance scores
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_texts([query], show_progress=False)[0]
        
        # Search vector database
        results = self.db.search(query_embedding, limit=limit * 3)  # Get more, then dedupe
        
        # Deduplicate by note (keep best chunk per note)
        note_results = {}
        for result in results:
            title = result["title"]
            
            # Calculate relevance score from distance
            distance = result.get("_distance", 0)
            relevance = max(0, 100 * (1 - distance / 2))  # Convert distance to 0-100 score
            
            if title not in note_results or relevance > note_results[title]["_relevance_score"]:
                note_results[title] = {
                    "title": title,
                    "creation_date": result["creation_date"],
                    "modification_date": result["modification_date"],
                    "_relevance_score": relevance,
                    "_best_chunk_index": result["chunk_index"],
                    "_total_chunks": result["total_chunks"],
                    "_matching_chunk_preview": result["chunk_content"][:300] + "..."
                }
        
        # Sort by relevance and limit
        sorted_results = sorted(
            note_results.values(),
            key=lambda x: x["_relevance_score"],
            reverse=True
        )[:limit]
        
        print(f"✅ Found {len(sorted_results)} relevant notes")
        return sorted_results


# ============================================================================
# MCP Server
# ============================================================================

app = Server("apple-notes-mcp-python")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="index-notes",
            description="Index Apple Notes for semantic search. Use 'fresh' mode to reindex all notes, 'incremental' to only process new/modified notes (default).",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fresh", "incremental"],
                        "default": "incremental",
                        "description": "Indexing mode"
                    }
                    ,"limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional: only process the first N notes from the Notes app"
                    }
                }
            }
        ),
        Tool(
            name="search-notes",
            description="Semantic search over your Apple Notes",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get-note",
            description="Get full content of a specific note by title",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title"
                    },
                    "creation_date": {
                        "type": "string",
                        "description": "Creation date (ISO format) if multiple notes have same title"
                    }
                },
                "required": ["title"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "index-notes":
            mode = arguments.get("mode", "incremental")
            limit = arguments.get("limit")
            indexer = NotesIndexer()
            result = await indexer.index_notes(mode=mode, limit=limit)
            
            message = f"Successfully indexed {result['processed']} notes into {result['chunks']} chunks in {result['time_seconds']:.1f}s\n\n"
            message += f"📊 Summary:\n"
            message += f"• Notes processed: {result['processed']}\n"
            message += f"• Chunks created: {result['chunks']}\n"
            
            if result.get('skipped', 0) > 0:
                message += f"• Skipped unchanged: {result['skipped']}\n"
            
            message += f"• Mode: {mode}\n"
            message += f"• Device: {DEVICE}\n\n"
            message += "✨ Your notes are ready for semantic search!"
            
            return [TextContent(type="text", text=message)]
        
        elif name == "search-notes":
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            
            searcher = NotesSearcher()
            results = searcher.search(query, limit=limit)
            
            if not results:
                return [TextContent(type="text", text=f"No results found for '{query}'")]
            
            message = f"Found {len(results)} relevant notes:\n\n"
            for i, result in enumerate(results, 1):
                message += f"{i}. **{result['title']}** (relevance: {result['_relevance_score']:.1f}%)\n"
                message += f"   📅 Created: {result['creation_date']}\n"
                message += f"   ✏️ Modified: {result['modification_date']}\n"
                message += f"   📄 Best match in chunk {result['_best_chunk_index'] + 1}/{result['_total_chunks']}\n"
                message += f"   Preview: {result['_matching_chunk_preview']}\n\n"
            
            return [TextContent(type="text", text=message)]
        
        elif name == "get-note":
            title = arguments["title"]
            creation_date = arguments.get("creation_date")
            
            extractor = AppleNotesExtractor()
            note = extractor.get_note_by_title_and_date(title, creation_date or "")
            
            if not note:
                return [TextContent(type="text", text=f"Note '{title}' not found")]
            
            processor = TextProcessor()
            plain_text = processor.html_to_plain_text(note.content)
            
            message = f"**{note.title}**\n\n"
            message += f"📅 Created: {note.creation_date}\n"
            message += f"✏️ Modified: {note.modification_date}\n\n"
            message += f"---\n\n{plain_text}"
            
            return [TextContent(type="text", text=message)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    print("🚀 Apple Notes MCP Server (Python Edition)")
    print(f"📊 Device: {DEVICE}")
    print(f"📦 Model: {MODEL_NAME}")
    asyncio.run(main())