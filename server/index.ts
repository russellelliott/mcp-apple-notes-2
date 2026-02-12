import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import * as lancedb from "@lancedb/lancedb";
import { runJxa } from "run-jxa";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
// Remove TurndownService import
import {
  EmbeddingFunction,
  LanceSchema,
  register,
} from "@lancedb/lancedb/embedding";
import { type Float, Float32, Utf8, FixedSizeList, Field } from "apache-arrow";
import { pipeline as hfPipeline } from "@huggingface/transformers";

// Remove the turndown instance
const db = await lancedb.connect(
  path.join(os.homedir(), ".mcp-apple-notes", "data")
);

// Path for notes cache file
const NOTES_CACHE_PATH = path.join(os.homedir(), ".mcp-apple-notes", "notes-cache.json");

// Types for note metadata
interface NoteMetadata {
  title: string;
  creation_date: string;
  modification_date: string;
}

interface NotesCache {
  last_sync: string;
  notes: NoteMetadata[];
}

interface ChunkData {
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
  chunk_index: string;
  total_chunks: string;
  chunk_content: string;
  cluster_id: string;
  cluster_label: string;
  cluster_confidence: string;
  cluster_summary: string;
  last_clustered: string;
}

// Utility functions for notes cache
const loadNotesCache = async (): Promise<NotesCache | null> => {
  try {
    const cacheContent = await fs.readFile(NOTES_CACHE_PATH, 'utf8');
    return JSON.parse(cacheContent);
  } catch (error) {
    console.log(`📁 No existing cache file found or error reading it`);
    return null;
  }
};

const saveNotesCache = async (notes: NoteMetadata[]): Promise<void> => {
  try {
    // Ensure directory exists
    const cacheDir = path.dirname(NOTES_CACHE_PATH);
    await fs.mkdir(cacheDir, { recursive: true });
    
    const cache: NotesCache = {
      last_sync: new Date().toISOString(),
      notes: notes
    };
    
    await fs.writeFile(NOTES_CACHE_PATH, JSON.stringify(cache, null, 2));
    console.log(`💾 Saved ${notes.length} notes to cache file`);
    console.log(`📅 Cache timestamp: ${cache.last_sync}`);
    
    // Show a sample of what's being cached
    if (notes.length > 0) {
      console.log(`📝 Sample cached notes:`);
      notes.slice(0, 3).forEach((note, idx) => {
        console.log(`   ${idx + 1}. "${note.title}" (created: ${note.creation_date}, modified: ${note.modification_date})`);
      });
      if (notes.length > 3) {
        console.log(`   ... and ${notes.length - 3} more notes`);
      }
    }
  } catch (error) {
    console.log(`⚠️ Failed to save cache file: ${(error as Error).message}`);
  }
};

// Helper to merge new notes with existing cache (for incremental updates)
const mergeNotesForCache = (newNotes: NoteMetadata[], existingCachedNotes: NoteMetadata[]): NoteMetadata[] => {
  // Create a map of existing notes by unique key (title + creation_date)
  const existingMap = new Map<string, NoteMetadata>();
  existingCachedNotes.forEach(note => {
    const key = `${note.title}|||${note.creation_date}`;
    existingMap.set(key, note);
  });
  
  // Add or update with new notes
  newNotes.forEach(note => {
    const key = `${note.title}|||${note.creation_date}`;
    existingMap.set(key, note); // This will update if exists, or add if new
  });
  
  // Convert back to array and sort by modification date (newest first)
  const mergedNotes = Array.from(existingMap.values());
  mergedNotes.sort((a, b) => {
    return new Date(b.modification_date).getTime() - new Date(a.modification_date).getTime();
  });
  
  return mergedNotes;
};

// Helper to identify new/modified notes
const identifyChangedNotes = (currentNotes: NoteMetadata[], cachedNotes: NoteMetadata[]): {
  newNotes: NoteMetadata[];
  modifiedNotes: NoteMetadata[];
  unchangedNotes: NoteMetadata[];
} => {
  const cachedMap = new Map<string, { creation_date: string; modification_date: string }>(); // title -> dates
  
  cachedNotes.forEach(note => {
    cachedMap.set(note.title, {
      creation_date: note.creation_date,
      modification_date: note.modification_date
    });
  });
  
  const newNotes: NoteMetadata[] = [];
  const modifiedNotes: NoteMetadata[] = [];
  const unchangedNotes: NoteMetadata[] = [];
  
  currentNotes.forEach(note => {
    const cached = cachedMap.get(note.title);
    
    if (!cached) {
      // New note (not in cache)
      newNotes.push(note);
    } else if (cached.modification_date !== note.modification_date) {
      // Modified note (modification date changed)
      modifiedNotes.push(note);
    } else if (cached.creation_date !== note.creation_date) {
      // Edge case: creation date changed (shouldn't happen but handle it)
      console.log(`⚠️ Note "${note.title}" has different creation date - treating as modified`);
      modifiedNotes.push(note);
    } else {
      // Unchanged note
      unchangedNotes.push(note);
    }
  });
  
  return { newNotes, modifiedNotes, unchangedNotes };
};

// HDBSCAN Clustering Functions removed

// Calculate semantic "quality score" for outlier assignment
// Evaluates how well an outlier semantically fits with a cluster
// QualityScore func removed

// reassignOutliers removed

// clusterRemainingOutliers removed

// clusterNotes removed

// getNotesInCluster removed

// listClusters removed

// generateClusterLabel removed

// Update to better embedding model
const extractor = await hfPipeline(
  "feature-extraction",
  "Xenova/bge-small-en-v1.5" // Better model for semantic search
);

// Get tokenizer from the model
const tokenizer = extractor.tokenizer;

// Chunking configuration
const CHUNK_SIZE = 400; // tokens (留出余量给系统 tokens)
const CHUNK_OVERLAP = 50; // tokens overlap between chunks
const MAX_CHUNK_SIZE = 512; // hard limit for safety

// Enhanced chunking function with better text preservation
const createChunks = async (text: string, maxTokens = CHUNK_SIZE, overlap = CHUNK_OVERLAP): Promise<string[]> => {
  if (!text || text.trim().length === 0) {
    return [''];
  }
  
  try {
    // First, try to estimate if we need chunking at all
    const roughTokenCount = text.length / 4; // Rough estimate: ~4 chars per token
    
    if (roughTokenCount <= maxTokens) {
      // Text is likely small enough, verify with actual tokenization
      const tokens = await tokenizer(text);
      const tokenIds = Array.from(tokens.input_ids.data);
      
      if (tokenIds.length <= maxTokens) {
        return [text]; // Return original text to preserve formatting
      }
    }
    
    // Text needs chunking - use a smarter approach
    // Split on natural boundaries first (paragraphs, sentences)
    const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    
    if (paragraphs.length === 1) {
      // Single paragraph, split on sentences
      const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
      return await createChunksFromSegments(sentences, maxTokens, overlap);
    } else {
      // Multiple paragraphs, try to chunk by paragraphs first
      return await createChunksFromSegments(paragraphs, maxTokens, overlap);
    }
    
  } catch (error) {
    console.log(`⚠️ Smart chunking failed, using fallback: ${error.message}`);
    return createFallbackChunks(text, maxTokens, overlap);
  }
};

// Helper function to create chunks from text segments (paragraphs or sentences)
const createChunksFromSegments = async (segments: string[], maxTokens: number, overlap: number): Promise<string[]> => {
  const chunks: string[] = [];
  let currentChunk = '';
  let currentTokens = 0;
  
  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    
    // Estimate tokens for this segment
    const segmentTokens = await estimateTokens(segment);
    
    // If adding this segment would exceed limit, finalize current chunk
    if (currentTokens + segmentTokens > maxTokens && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      
      // Start new chunk with overlap
      const overlapText = createOverlapText(currentChunk, overlap);
      currentChunk = overlapText + (overlapText ? '\n\n' : '') + segment;
      currentTokens = await estimateTokens(currentChunk);
    } else {
      // Add segment to current chunk
      if (currentChunk.length > 0) {
        currentChunk += '\n\n' + segment;
      } else {
        currentChunk = segment;
      }
      currentTokens += segmentTokens;
    }
    
    // If a single segment is too large, split it further
    if (segmentTokens > maxTokens) {
      chunks.push(...createFallbackChunks(segment, maxTokens, overlap));
      currentChunk = '';
      currentTokens = 0;
    }
  }
  
  // Add final chunk
  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks.length > 0 ? chunks : [segments.join('\n\n')];
};

// Helper to estimate token count without full tokenization
const estimateTokens = async (text: string): Promise<number> => {
  // For performance, use character-based estimation for most cases
  const charEstimate = Math.ceil(text.length / 4);
  
  // If it's close to the limit, do actual tokenization
  if (charEstimate > CHUNK_SIZE * 0.8) {
    try {
      const tokens = await tokenizer(text);
      return tokens.input_ids.data.length;
    } catch {
      return charEstimate;
    }
  }
  
  return charEstimate;
};

// Helper to create overlap text from the end of previous chunk
const createOverlapText = (chunk: string, overlapTokens: number): string => {
  if (!chunk || overlapTokens <= 0) return '';
  
  // Take approximately the last portion for overlap
  const overlapChars = overlapTokens * 4; // Rough estimate
  const words = chunk.split(/\s+/);
  
  // Take last few words to approximate overlap
  const overlapWords = words.slice(-Math.max(1, Math.floor(overlapTokens / 2)));
  return overlapWords.join(' ');
};

// Fallback chunking using character-based approach (preserves formatting better)
const createFallbackChunks = (text: string, maxTokens: number, overlap: number): string[] => {
  const approxChunkSize = maxTokens * 4; // ~4 chars per token
  const approxOverlap = overlap * 4;
  
  if (text.length <= approxChunkSize) {
    return [text];
  }
  
  const chunks: string[] = [];
  let start = 0;
  
  while (start < text.length) {
    const end = Math.min(start + approxChunkSize, text.length);
    let chunk = text.substring(start, end);
    
    // Try to break on word boundaries
    if (end < text.length) {
      const lastSpace = chunk.lastIndexOf(' ');
      const lastNewline = chunk.lastIndexOf('\n');
      const breakPoint = Math.max(lastSpace, lastNewline);
      
      if (breakPoint > start + approxChunkSize * 0.7) {
        chunk = text.substring(start, start + breakPoint);
        start = start + breakPoint + 1;
      } else {
        start = end;
      }
    } else {
      start = end;
    }
    
    if (chunk.trim().length > 0) {
      chunks.push(chunk.trim());
    }
    
    // Apply overlap for next chunk
    if (start < text.length) {
      start = Math.max(start - approxOverlap, 0);
    }
  }
  
  return chunks.length > 0 ? chunks : [text.substring(0, approxChunkSize)];
};

@register("openai")
export class OnDeviceEmbeddingFunction extends EmbeddingFunction<string> {
  toJSON(): object {
    return {};
  }
  ndims() {
    return 384; // bge-small-en-v1.5 uses 384 dimensions
  }
  embeddingDataType(): Float {
    return new Float32();
  }
  
  // Enhanced preprocessing for better semantic capture
  private cleanText(text: string): string {
    return text
      .toLowerCase() // Normalize case
      .replace(/\s+/g, ' ') // Normalize whitespace
      .replace(/[^\w\s\-.,!?;:()\[\]{}'"]/g, ' ') // Keep basic punctuation
      .replace(/\s+/g, ' ') // Clean up extra spaces
      .trim();
  }
  
  async computeQueryEmbeddings(data: string) {
    const cleanedData = this.cleanText(data);
    const output = await extractor(cleanedData, { 
      pooling: "mean", 
      normalize: true // Critical for proper similarity calculation
    });
    return output.data as number[];
  }
  
  async computeSourceEmbeddings(data: string[]) {
    // Process embeddings in batches for better performance
    const EMBEDDING_BATCH_SIZE = 10;
    const results = [];
    
    for (let i = 0; i < data.length; i += EMBEDDING_BATCH_SIZE) {
      const batch = data.slice(i, i + EMBEDDING_BATCH_SIZE);
      const batchResults = await Promise.all(
        batch.map(async (item) => {
          const cleanedItem = this.cleanText(item);
          const output = await extractor(cleanedItem, { 
            pooling: "mean", 
            normalize: true
          });
          return output.data as number[];
        })
      );
      results.push(...batchResults);
    }
    
    return results;
  }
}




//convert html to plaintext
// Replace the HTML to text conversion function
const htmlToPlainText = (html: string): string => {
  if (!html) return "";
  
  return html
    // Remove script and style elements completely
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    
    // Convert common HTML elements to readable text
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<\/h[1-6]>/gi, '\n\n')
    .replace(/<\/li>/gi, '\n')
    .replace(/<\/tr>/gi, '\n')
    .replace(/<\/td>/gi, ' | ')
    
    // Handle lists
    .replace(/<ul[^>]*>/gi, '\n')
    .replace(/<\/ul>/gi, '\n')
    .replace(/<ol[^>]*>/gi, '\n')
    .replace(/<\/ol>/gi, '\n')
    .replace(/<li[^>]*>/gi, '• ')
    
    // Handle headers - preserve their content but make them readable
    .replace(/<h1[^>]*>(.*?)<\/h1>/gi, '\n\n$1\n' + '='.repeat(50) + '\n')
    .replace(/<h2[^>]*>(.*?)<\/h2>/gi, '\n\n$1\n' + '-'.repeat(30) + '\n')
    .replace(/<h3[^>]*>(.*?)<\/h3>/gi, '\n\n$1\n')
    .replace(/<h[4-6][^>]*>(.*?)<\/h[4-6]>/gi, '\n\n$1\n')
    
    // Handle emphasis
    .replace(/<strong[^>]*>(.*?)<\/strong>/gi, '**$1**')
    .replace(/<b[^>]*>(.*?)<\/b>/gi, '**$1**')
    .replace(/<em[^>]*>(.*?)<\/em>/gi, '*$1*')
    .replace(/<i[^>]*>(.*?)<\/i>/gi, '*$1*')
    
    // Handle links
    .replace(/<a[^>]*href=["']([^"']+)["'][^>]*>(.*?)<\/a>/gi, '$2 ($1)')
    
    // Remove all remaining HTML tags
    .replace(/<[^>]*>/g, '')
    
    // Clean up entities
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&[a-zA-Z]+;/g, '') // Remove other entities
    
    // Clean up whitespace
    .replace(/\n\s*\n\s*\n/g, '\n\n') // Max 2 consecutive newlines
    .replace(/[ \t]+/g, ' ') // Multiple spaces/tabs to single space
    .trim();
};

// Initialize the embedding function first
const func = new OnDeviceEmbeddingFunction();
console.log('🤖 Initialized embedding function');

// Updated schema to include chunk information and clustering fields
const notesTableSchema = LanceSchema({
  title: new Utf8(), // Regular field, not for embedding
  content: new Utf8(), // Regular field, not for embedding  
  creation_date: new Utf8(), // Regular field
  modification_date: new Utf8(), // Regular field
  chunk_index: new Utf8(), // Regular field
  total_chunks: new Utf8(), // Regular field
  chunk_content: func.sourceField(new Utf8()), // This is the field that gets embedded
  vector: func.vectorField(), // This stores the embeddings
  
  // NEW: Clustering fields (same value for all chunks of the same note)
  cluster_id: new Utf8(), // -1 for outliers, 0+ for cluster ID (using string to handle -1)
  cluster_label: new Utf8(), // Human-readable name like "Work Projects", "Python Development"
  cluster_confidence: new Utf8(), // How strongly this note belongs to the cluster (0.0-1.0)
  cluster_summary: new Utf8(), // Auto-generated description of what this cluster contains
  last_clustered: new Utf8(), // ISO timestamp when clustering was last run
});

const QueryNotesSchema = z.object({
  query: z.string(),
});

const GetNoteSchema = z.object({
  title: z.string(),
  creation_date: z.string().optional(),
});

const IndexNotesSchema = z.object({
  mode: z.enum(["fresh", "incremental"]).optional().default("incremental"),
});

export const server = new Server(
  {
    name: "my-apple-notes-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Add a shutdown method
export const shutdown = async () => {
  await db.close();
  // Force cleanup of the pipeline
  if (extractor) {
    // @ts-ignore - accessing internal cleanup method
    await extractor?.cleanup?.();
  }
  // Force exit since stdio transport doesn't have cleanup
  process.exit(0);
};

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list-notes",
        description: "Lists just the titles of all my Apple Notes",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "index-notes",
        description:
          "Index all my Apple Notes for Semantic Search using enhanced method that handles duplicate note titles better. Uses incremental mode by default to only process new/modified notes. Please tell the user that the sync takes couple of seconds up to couple of minutes depending on how many notes you have.",
        inputSchema: {
          type: "object",
          properties: {
            mode: {
              type: "string",
              enum: ["fresh", "incremental"],
              description: "fresh: reindex all notes from scratch, incremental: only process new/modified notes",
              default: "incremental"
            }
          },
          required: [],
        },
      },
      {
        name: "get-note",
        description: "Get a note full content and details by title. If multiple notes have the same title, you can specify creation_date to get a specific one.",
        inputSchema: {
          type: "object",
          properties: {
            title: z.string(),
            creation_date: z.string().optional(),
          },
          required: ["title"],
        },
      },
      {
        name: "search-notes",
        description: "Search for notes by title or content",
        inputSchema: {
          type: "object",
          properties: {
            query: z.string(),
          },
          required: ["query"],
        },
      },
// Clustering tools removed
      {
        name: "create-note",
        description:
          "Create a new Apple Note with specified title and content. Must be in HTML format WITHOUT newlines",
        inputSchema: {
          type: "object",
          properties: {
            title: { type: "string" },
            content: { type: "string" },
          },
          required: ["title", "content"],
        },
      }
    ],
  };
});

const getNotes = async function* (maxNotes?: number) {
  console.log("   Requesting notes list from Apple Notes...");
  try {
    const BATCH_SIZE = 50; // Increased from 25 to 50 for faster note fetching
    let startIndex = 1;
    let hasMore = true;

    // Get total count or use the limit
    let totalCount: number;
    
    if (maxNotes) {
      totalCount = maxNotes;
      console.log(`   🎯 Using subset limit: ${totalCount} notes`);
    } else {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 120000);

      totalCount = await Promise.race([
        runJxa(`
          const app = Application('Notes');
          app.includeStandardAdditions = true;
          return app.notes().length;
        `),
        new Promise((_, reject) => {
          controller.signal.addEventListener('abort', () => 
            reject(new Error('Getting notes count timed out after 120s'))
          );
        })
      ]) as number;

      clearTimeout(timeout);
      console.log(`   📊 Total notes found: ${totalCount}`);
    }

    while (hasMore) {
      console.log(`   Fetching batch of notes (${startIndex} to ${startIndex + BATCH_SIZE - 1})...`);
      
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 120000);

      const batchResult = await Promise.race([
        runJxa(`
          const app = Application('Notes');
          app.includeStandardAdditions = true;
          
          const titles = [];
          for (let i = ${startIndex}; i < ${startIndex + BATCH_SIZE}; i++) {
            try {
              const note = app.notes[i - 1];
              if (note) {
                titles.push(note.name());
              }
            } catch (error) {
              continue;
            }
          }
          return titles;
        `),
        new Promise((_, reject) => {
          controller.signal.addEventListener('abort', () => 
            reject(new Error('Getting notes batch timed out after 120s'))
          );
        })
      ]);

      clearTimeout(timeout);
      
      const titles = batchResult as string[];
      
      // Yield the batch along with progress info
      yield {
        titles,
        progress: {
          current: startIndex + titles.length - 1,
          total: totalCount,
          batch: {
            start: startIndex,
            end: startIndex + BATCH_SIZE - 1
          }
        }
      };
      
      startIndex += BATCH_SIZE;
      hasMore = startIndex <= totalCount && titles.length > 0;

      await new Promise(resolve => setTimeout(resolve, 500)); // Reduced from 1000ms to 500ms
    }

  } catch (error) {
    console.error("   ❌ Error getting notes list:", error.message);
    throw new Error(`Failed to get notes list: ${error.message}`);
  }
};

// Update the existing getNoteDetailsByTitle to use the new function
const getNoteDetailsByTitle = async (title: string, creationDate?: string) => {
  // If creation date is provided, fetch that specific note
  if (creationDate) {
    const note = await getNoteByTitleAndDate(title, creationDate);
    if (!note) {
      throw new Error(`Note "${title}" with creation date "${creationDate}" not found`);
    }
    return note;
  }
  
  // Otherwise, find all notes with this title
  const notesWithTitle = await runJxa(`
    const app = Application('Notes');
    const targetTitle = "${title.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}";
    
    try {
      const matchingNotes = app.notes.whose({name: targetTitle});
      const results = [];
      
      for (let i = 0; i < matchingNotes.length; i++) {
        const note = matchingNotes[i];
        results.push({
          title: note.name(),
          creation_date: note.creationDate().toLocaleString()
        });
      }
      
      return JSON.stringify(results);
    } catch (error) {
      return "[]";
    }
  `);
  
  const matches = JSON.parse(notesWithTitle as string) as Array<{
    title: string;
    creation_date: string;
  }>;
  
  if (matches.length === 0) {
    throw new Error(`Note "${title}" not found`);
  }
  
  if (matches.length === 1) {
    // Single note, fetch it
    return await getNoteByTitleAndDate(matches[0].title, matches[0].creation_date);
  }
  
  // Multiple notes with same title - return info about all of them
  throw new Error(
    `Multiple notes found with title "${title}".\n` +
    `Found ${matches.length} notes with creation dates:\n` +
    matches.map((m, i) => `  ${i + 1}. Created: ${m.creation_date}`).join('\n') +
    `\n\nPlease specify the creation_date parameter to get a specific note.`
  );
};

// New helper function to get note by title AND creation date
const getNoteByTitleAndDate = async (title: string, creationDate: string, customTimeout?: number) => {
  const NOTE_FETCH_TIMEOUT = customTimeout || 300000; // Default 5 minutes per note
  
  // Escape special characters in title and date
  const escapedTitle = title.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escapedDate = creationDate.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  
  // Add timeout to prevent hanging on problematic notes
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    console.log(`⏰ Timeout: Note "${title}" is taking too long (>${NOTE_FETCH_TIMEOUT/1000}s), skipping...`);
    controller.abort();
  }, NOTE_FETCH_TIMEOUT);
  
  try {
    const note = await Promise.race([
      runJxa(`
    const app = Application('Notes');
    const targetTitle = "${escapedTitle}";
    const targetDate = "${escapedDate}";
    
    try {
      // Get all notes with matching title
      const matchingNotes = app.notes.whose({name: targetTitle});
      
      if (matchingNotes.length === 0) {
        return "{}";
      }
      
      // If only one note with this title, return it
      if (matchingNotes.length === 1) {
        const note = matchingNotes[0];
        return JSON.stringify({
          title: note.name(),
          content: note.body(),
          creation_date: note.creationDate().toLocaleString(),
          modification_date: note.modificationDate().toLocaleString()
        });
      }
      
      // Multiple notes with same title - find by creation date
      for (let i = 0; i < matchingNotes.length; i++) {
        const note = matchingNotes[i];
        const noteDate = note.creationDate().toLocaleString();
        
        if (noteDate === targetDate) {
          return JSON.stringify({
            title: note.name(),
            content: note.body(),
            creation_date: noteDate,
            modification_date: note.modificationDate().toLocaleString()
          });
        }
      }
      
      // Fallback: return first match if date doesn't match exactly
      // (date formatting might differ slightly)
      const note = matchingNotes[0];
      return JSON.stringify({
        title: note.name(),
        content: note.body(),
        creation_date: note.creationDate().toLocaleString(),
        modification_date: note.modificationDate().toLocaleString()
      });
      
    } catch (error) {
      console.log("Error fetching note: " + error.toString());
      return "{}";
    }
  `),
      new Promise((_, reject) => {
        controller.signal.addEventListener('abort', () =>
          reject(new Error(`Note fetch timeout after ${NOTE_FETCH_TIMEOUT/1000} seconds: "${title}"`))
        );
      })
    ]);

    clearTimeout(timeout);
    const parsed = JSON.parse(note as string);
    
    // Return null if empty object (note not found)
    if (Object.keys(parsed).length === 0) {
      return null;
    }
    
    return parsed as {
      title: string;
      content: string;
      creation_date: string;
      modification_date: string;
    };
    
  } catch (error) {
    clearTimeout(timeout);
    
    // Check if it was a timeout error
    if ((error as Error).message.includes('timeout')) {
      console.log(`⏰ Skipped note "${title}" due to timeout (>5min)`);
      return null;
    }
    
    // Other errors
    console.log(`❌ Error fetching note "${title}": ${(error as Error).message}`);
    return null;
  }
};

// Enhanced fetchAndIndexAllNotes function that fetches by title and creation date
export const fetchAndIndexAllNotes = async (notesTable: any, maxNotes?: number, mode: 'fresh' | 'incremental' | 'incremental-since' = 'incremental') => {
  const start = performance.now();
  
  console.log(`Starting notes fetch and indexing${maxNotes ? ` (max: ${maxNotes} notes)` : ''} in ${mode} mode...`);

  // Step 0: For incremental-since mode, determine the threshold date
  let thresholdDate: number | null = null;
  if (mode === 'incremental-since') {
    console.log('📅 determining threshold date for incremental-since mode...');
    try {
      const dbChunks = await notesTable.query()
        .limit(100000)
        .select(["modification_date"])
        .toArray();
      
      for (const chunk of dbChunks) {
        const t = new Date(chunk.modification_date).getTime();
        if (thresholdDate === null || t > thresholdDate) {
          thresholdDate = t;
        }
      }
      
      if (thresholdDate !== null) {
        console.log(`📅 Threshold found: ${new Date(thresholdDate).toLocaleString()} (will stop fetching older notes)`);
      } else {
        console.log(`ℹ️ No existing notes found, fetching all.`);
      }
    } catch (error) {
      console.log(`⚠️ Could not determine threshold date: ${(error as Error).message}. Fetching all.`);
    }
  }
  
  // Step 1: First fetch all titles, creation dates, and modification dates
  console.log('\nStep 1: Fetching note titles, creation dates, and modification dates...');
  
  // First get the total count quickly
  console.log('📊 Getting total note count...');
  const totalNotesCount = await runJxa(`
    const app = Application('Notes');
    return app.notes().length;
  `) as number;
  
  const limitCount = maxNotes ? Math.min(totalNotesCount, maxNotes) : totalNotesCount;
  console.log(`📋 Found ${totalNotesCount} notes${maxNotes ? `, limiting to ${limitCount}` : ''}`);
  
  // Process notes in batches with progress updates
  const TITLE_BATCH_SIZE = 50;
  const allNoteTitles: Array<{
    title: string;
    creation_date: string;
    modification_date: string;
  }> = [];
  
  let titleProgress = 0;
  const totalTitleBatches = Math.ceil(limitCount / TITLE_BATCH_SIZE);
  
  console.log(`🔄 Processing titles in ${totalTitleBatches} batches of ${TITLE_BATCH_SIZE}...`);
  
  for (let batchStart = 0; batchStart < limitCount; batchStart += TITLE_BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + TITLE_BATCH_SIZE, limitCount);
    const batchNum = Math.floor(batchStart / TITLE_BATCH_SIZE) + 1;
    
    console.log(`📦 [${batchNum}/${totalTitleBatches}] Fetching titles ${batchStart + 1}-${batchEnd}...`);
    
    const batchTitlesData = await runJxa(`
      const app = Application('Notes');
      const notes = app.notes();
      const startIdx = ${batchStart};
      const endIdx = ${batchEnd};
      const noteTitles = [];
      
      for (let i = startIdx; i < endIdx; i++) {
        try {
          const note = notes[i];
          noteTitles.push({
            title: note.name(),
            creation_date: note.creationDate().toLocaleString(),
            modification_date: note.modificationDate().toLocaleString()
          });
        } catch (error) {
          // Skip problematic notes
          continue;
        }
      }
      
      return JSON.stringify(noteTitles);
    `);
    
    let batchTitles = JSON.parse(batchTitlesData as string) as Array<{
      title: string;
      creation_date: string;
      modification_date: string;
    }>;
    
    // For incremental-since mode, check if we've reached older notes
    if (thresholdDate !== null) {
      // Find the first note that is older than or equal to the threshold
      const cutoffIndex = batchTitles.findIndex(n => new Date(n.modification_date).getTime() <= thresholdDate!);
      
      if (cutoffIndex !== -1) {
        console.log(`🛑 Found note overlapping with threshold at index ${cutoffIndex} in batch. Stopping fetch.`);
        // Take only the new notes
        batchTitles = batchTitles.slice(0, cutoffIndex);
        allNoteTitles.push(...batchTitles);
        console.log(`✅ [${batchNum}/${totalTitleBatches}] Got ${batchTitles.length} titles (truncated)`);
        break; // Stop fetching more batches
      }
    }
    
    allNoteTitles.push(...batchTitles);
    titleProgress = batchEnd;
    
    console.log(`✅ [${batchNum}/${totalTitleBatches}] Got ${batchTitles.length} titles (${titleProgress}/${limitCount} total)`);
  }
  
  console.log(`✨ Fetched ${allNoteTitles.length} note titles in ${((performance.now() - start)/1000).toFixed(1)}s`);
  
  const noteTitles = allNoteTitles;
  
  // Step 2: Determine which notes to process based on mode
  let notesToProcess: NoteMetadata[] = noteTitles;
  let skippedCount = 0;
  
  if (mode === 'incremental' || mode === 'incremental-since') {
    console.log(`\nStep 2: Comparing with cached notes to find changes (${mode})...`);
    
    let cachedNotesData: NoteMetadata[] = [];
    
    // Use database as source of truth instead of external cache file
    try {
      console.log(`📂 Loading existing notes from database...`);
      
      // Query all unique notes from database (get one chunk per note to extract metadata)
      const dbChunks = await notesTable.query()
        .limit(100000)
        .select(["title", "creation_date", "modification_date"])
        .toArray();
      
      // Deduplicate by title + creation_date (since same note can have multiple chunks)
      const noteMap = new Map<string, NoteMetadata>();
      dbChunks.forEach(chunk => {
        const key = `${chunk.title}|||${chunk.creation_date}`;
        if (!noteMap.has(key)) {
          noteMap.set(key, {
            title: chunk.title,
            creation_date: chunk.creation_date,
            modification_date: chunk.modification_date
          });
        }
      });
      
      cachedNotesData = Array.from(noteMap.values());
      console.log(`📂 Found ${cachedNotesData.length} existing notes in database`);
      
    } catch (dbError) {
      console.log(`⚠️ Could not load from database: ${(dbError as Error).message}`);
      console.log(`📂 Trying JSON cache file as fallback...`);
      
      // Fallback to JSON cache if database query fails
      try {
        const cachedNotes = await loadNotesCache();
        if (cachedNotes) {
          cachedNotesData = cachedNotes.notes;
          console.log(`📂 Found cache with ${cachedNotesData.length} notes from ${cachedNotes.last_sync}`);
        }
      } catch (cacheError) {
        console.log(`ℹ️ No existing cache found, treating all notes as new`);
      }
    }
    
    if (cachedNotesData.length > 0) {
      const { newNotes, modifiedNotes, unchangedNotes } = identifyChangedNotes(noteTitles, cachedNotesData);
      
      console.log(`📊 Change analysis:`);
      console.log(`  • New notes: ${newNotes.length}`);
      console.log(`  • Modified notes: ${modifiedNotes.length}`);
      console.log(`  • Unchanged notes: ${unchangedNotes.length}`);
      
      // Show details of new notes
      if (newNotes.length > 0) {
        console.log(`\n🆕 New notes detected:`);
        newNotes.slice(0, 10).forEach((note, idx) => {
          console.log(`  ${idx + 1}. "${note.title}" (created: ${note.creation_date}, modified: ${note.modification_date})`);
        });
        if (newNotes.length > 10) {
          console.log(`  ... and ${newNotes.length - 10} more new notes`);
        }
      }
      
      // Show details of modified notes
      if (modifiedNotes.length > 0) {
        console.log(`\n✏️ Modified notes detected:`);
        modifiedNotes.slice(0, 10).forEach((note, idx) => {
          const cached = cachedNotesData.find(c => c.title === note.title);
          console.log(`  ${idx + 1}. "${note.title}"`);
          console.log(`      Created: ${note.creation_date}`);
          console.log(`      Modified: ${cached?.modification_date} → ${note.modification_date}`);
        });
        if (modifiedNotes.length > 10) {
          console.log(`  ... and ${modifiedNotes.length - 10} more modified notes`);
        }
      }
      
      notesToProcess = [...newNotes, ...modifiedNotes];
      skippedCount = unchangedNotes.length;
      
      if (notesToProcess.length === 0) {
        console.log(`✨ No changes detected! All notes are up to date.`);
        // Still save the cache to update last_sync time
        await saveNotesCache(noteTitles);
        return { processed: 0, totalChunks: 0, failed: 0, skipped: skippedCount, timeSeconds: (performance.now() - start) / 1000 };
      }
      
      // Remove old chunks for modified notes from database
      if (modifiedNotes.length > 0) {
        console.log(`\n🗑️ Removing old chunks for ${modifiedNotes.length} modified notes...`);
        for (const modNote of modifiedNotes) {
          try {
            // Delete existing chunks for this note
            await notesTable.delete(`title = '${modNote.title.replace(/'/g, "''")}'`);
            console.log(`   ✅ Removed old chunks for "${modNote.title}"`);
          } catch (error) {
            console.log(`   ⚠️ Could not remove old chunks for "${modNote.title}": ${(error as Error).message}`);
          }
        }
      }
    } else {
      console.log(`📁 No cache found, processing all ${noteTitles.length} notes`);
    }
  } else {
    console.log(`\nStep 2: Fresh mode - processing all ${noteTitles.length} notes`);
  }
  
  // Step 3: Process notes in batches - fetch, chunk, and immediately write to database
  console.log(`\nStep 3: Processing ${notesToProcess.length} notes in memory-efficient batches...`);
  console.log(`💡 Each batch will be: fetched → chunked → written to database immediately`);
  console.log(`📈 This approach minimizes memory usage by not storing all notes/chunks in memory at once\n`);
  
  let totalChunks = 0;
  let totalProcessed = 0;
  let totalFailed = 0;
  let totalTimeouts = 0;
  let totalRetries = 0;
  const batchSize = 50; // Process in batches for better performance
  const DB_BATCH_SIZE = 100; // Chunks per database write
  
  // Track notes that timed out for retry later
  const timedOutNotes: Array<{
    title: string;
    creation_date: string;
    modification_date: string;
    attempt: number;
  }> = [];
  
  // Get initial row count for verification
  const initialRowCount = await notesTable.countRows();
  console.log(`📊 Initial database rows: ${initialRowCount}`);
  
  // Process each batch independently to minimize memory usage
  for (let i = 0; i < notesToProcess.length; i += batchSize) {
    const batch = notesToProcess.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(notesToProcess.length / batchSize);
    
    console.log(`\n📦 Processing batch ${batchNum}/${totalBatches} (${batch.length} notes):`);
    
    // Step 3a: Fetch batch content in parallel
    console.log(`   📥 Fetching content for batch ${batchNum}... (timeout: 5min per note)`);
    const batchResults = await Promise.all(
      batch.map(async ({ title, creation_date, modification_date }, index) => {
        try {
          console.log(`     📄 [${batchNum}.${index + 1}] Fetching: "${title}"`);
          const start = performance.now();
          const result = await getNoteByTitleAndDate(title, creation_date);
          const duration = (performance.now() - start) / 1000;
          
          if (result) {
            console.log(`     ✅ [${batchNum}.${index + 1}] Success: "${title}" (${duration.toFixed(1)}s)`);
            return {
              title: result.title,
              content: result.content,
              creation_date: result.creation_date,
              modification_date: modification_date, // Use the fresh modification date
              _fetchDuration: duration
            };
          } else {
            // Check if this was likely a timeout (took close to 5 minute timeout)
            if (duration >= 299) { // Close to 300 second timeout
              console.log(`     ⏰ [${batchNum}.${index + 1}] Likely timeout: "${title}" (${duration.toFixed(1)}s, will retry)`);
              timedOutNotes.push({
                title,
                creation_date,
                modification_date,
                attempt: 1
              });
              totalTimeouts++;
            } else {
              console.log(`     ⚠️ [${batchNum}.${index + 1}] Empty result: "${title}" (${duration.toFixed(1)}s)`);
            }
          }
          return null;
        } catch (error) {
          const errorMsg = (error as Error).message;
          if (errorMsg.includes('timeout') || errorMsg.includes('Timeout')) {
            console.log(`     ⏰ [${batchNum}.${index + 1}] Timeout: "${title}" (will retry later with longer timeout after 5min)`);
            timedOutNotes.push({
              title,
              creation_date,
              modification_date,
              attempt: 1
            });
            totalTimeouts++;
          } else {
            console.log(`     ❌ [${batchNum}.${index + 1}] Failed: "${title}" - ${errorMsg}`);
          }
          return null;
        }
      })
    );
    
    const successfulNotes = batchResults.filter(note => note !== null);
    console.log(`   📊 Fetched: ${successfulNotes.length}/${batch.length} notes successfully`);
    
    // Step 3b: Process batch into chunks
    console.log(`   ✂️ Processing ${successfulNotes.length} notes into chunks...`);
    const batchChunks: ChunkData[] = [];
    let batchProcessed = 0;
    let batchFailed = 0;
    
    for (const note of successfulNotes) {
      try {
        const plainText = htmlToPlainText(note.content || "");
        const fullText = `${note.title}\n\n${plainText}`;
        const chunks = await createChunks(fullText);
        
        chunks.forEach((chunkContent, index) => {
          batchChunks.push({
            title: note.title,
            content: plainText,
            creation_date: note.creation_date,
            modification_date: note.modification_date,
            chunk_index: index.toString(),
            total_chunks: chunks.length.toString(),
            chunk_content: chunkContent,
            // Initialize cluster fields as empty - will be populated when clustering is run
            cluster_id: "",
            cluster_label: "",
            cluster_confidence: "",
            cluster_summary: "",
            last_clustered: "",
          });
        });
        
        batchProcessed++;
        console.log(`     📝 [${batchProcessed}/${successfulNotes.length}] "${note.title}" → ${chunks.length} chunks`);
        
      } catch (error) {
        batchFailed++;
        console.log(`     ❌ [${batchProcessed + batchFailed}/${successfulNotes.length}] Failed to chunk "${note.title}": ${(error as Error).message}`);
      }
    }
    
    console.log(`   📊 Batch ${batchNum} chunks: ${batchChunks.length} total from ${batchProcessed} notes`);
    
    // Step 3c: Write batch chunks to database immediately
    if (batchChunks.length > 0) {
      console.log(`   💾 Writing ${batchChunks.length} chunks to database...`);
      
      // Write chunks in sub-batches for optimal database performance
      const chunkBatches = Math.ceil(batchChunks.length / DB_BATCH_SIZE);
      for (let j = 0; j < batchChunks.length; j += DB_BATCH_SIZE) {
        const chunkBatch = batchChunks.slice(j, j + DB_BATCH_SIZE);
        const chunkBatchNum = Math.floor(j / DB_BATCH_SIZE) + 1;
        
        try {
          console.log(`     🔄 [${chunkBatchNum}/${chunkBatches}] Adding ${chunkBatch.length} chunks...`);
          
          // Check if we need manual embeddings due to missing embedding function metadata
          if ((notesTable as any)._needsManualEmbeddings) {
            console.log(`     🔧 Using manual embeddings (broken metadata mode)...`);
            
            // Generate embeddings manually and convert to Float32Array
            const processedBatch = await Promise.all(chunkBatch.map(async (chunk) => {
              const embeddings = await func.computeSourceEmbeddings([chunk.chunk_content]);
              // Convert to regular Array to avoid "vector.0" schema inference errors
              // where Float32Array is treated as a struct/object instead of a list
              return {
                ...chunk,
                vector: Array.from(embeddings[0])
              };
            }));
            
            // DO NOT pass embeddings parameter - we already have the vectors
            // Passing embeddings causes LanceDB to try to validate against broken metadata
            await notesTable.add(processedBatch);
            
          } else {
            // Normal automatic mode
            await notesTable.add(chunkBatch, { 
              embeddings: {
                chunk_content: func
              }
            });
          }
          
          console.log(`     ✅ [${chunkBatchNum}/${chunkBatches}] Wrote ${chunkBatch.length} chunks to database`);
        } catch (error) {
          console.error(`     ❌ [${chunkBatchNum}/${chunkBatches}] Failed to write chunk batch:`, error);
          throw error;
        }
      }
      
      // Verify database write
      const currentRowCount = await notesTable.countRows();
      console.log(`   🔍 Database now has ${currentRowCount} total rows (+${batchChunks.length} from this batch)`);
    }
    
    // Update totals
    totalChunks += batchChunks.length;
    totalProcessed += batchProcessed;
    totalFailed += batchFailed;
    
    console.log(`✅ Batch ${batchNum}/${totalBatches} complete: ${batchProcessed} notes → ${batchChunks.length} chunks written to database`);
    console.log(`📊 Overall progress: ${totalProcessed}/${notesToProcess.length} notes processed, ${totalChunks} total chunks`);
    
    // Progressive cache saving - save cache after every few batches as backup
    if (batchNum % 5 === 0 || batchNum === totalBatches) {
      console.log(`   💾 Saving progress to backup cache (batch ${batchNum}/${totalBatches})...`);
      try {
        // Merge with existing backup cache
        const existingCache = await loadNotesCache();
        const existingNotes = existingCache?.notes || [];
        const mergedNotes = mergeNotesForCache(allNoteTitles, existingNotes);
        await saveNotesCache(mergedNotes);
        console.log(`   ✅ Backup cache updated successfully`);
      } catch (error) {
        console.log(`   ⚠️ Backup cache save failed (non-critical): ${(error as Error).message}`);
      }
    }
    
    // Clear batch data from memory before next iteration
    // (This happens automatically with block scope, but being explicit)
  }
  
  // Step 3d: Retry timed-out notes with progressively longer timeouts
  console.log(`\n🔍 Timeout Summary: ${timedOutNotes.length} notes timed out during batch processing`);
  if (timedOutNotes.length > 0) {
    console.log(`\n🔄 Retrying ${timedOutNotes.length} timed-out notes with longer timeouts...`);
    timedOutNotes.forEach((note, i) => {
      console.log(`   ${i + 1}. "${note.title}" (attempt ${note.attempt})`);
    });
    
    const maxRetries = 2; // Try up to 2 additional times
    const timeoutMultipliers = [1.5, 2]; // 7.5min, then 10min
    
    for (let retryRound = 0; retryRound < maxRetries && timedOutNotes.length > 0; retryRound++) {
      const currentTimeout = 300000 * timeoutMultipliers[retryRound];
      console.log(`\n🕐 Retry round ${retryRound + 1}/${maxRetries}: ${timedOutNotes.length} notes with ${currentTimeout/1000}s timeout`);
      
      const retryResults = [];
      const stillTimedOut = [];
      
      // Process retries one at a time to avoid overwhelming the system
      for (let i = 0; i < timedOutNotes.length; i++) {
        const note = timedOutNotes[i];
        console.log(`   🔄 [${i + 1}/${timedOutNotes.length}] Retry "${note.title}" (${currentTimeout/1000}s timeout)`);
        
        try {
          const start = performance.now();
          const result = await getNoteByTitleAndDate(note.title, note.creation_date, currentTimeout);
          const duration = (performance.now() - start) / 1000;
          
          if (result) {
            console.log(`   ✅ [${i + 1}/${timedOutNotes.length}] Retry success: "${note.title}" (${duration.toFixed(1)}s)`);
            retryResults.push({
              title: result.title,
              content: result.content,
              creation_date: result.creation_date,
              modification_date: note.modification_date,
              _fetchDuration: duration,
              _wasRetry: true
            });
            totalRetries++;
          } else {
            console.log(`   ⚠️ [${i + 1}/${timedOutNotes.length}] Retry empty result: "${note.title}" (${duration.toFixed(1)}s)`);
          }
        } catch (error) {
          const errorMsg = (error as Error).message;
          if (errorMsg.includes('timeout')) {
            console.log(`   ⏰ [${i + 1}/${timedOutNotes.length}] Retry timeout: "${note.title}" (still too slow)`);
            stillTimedOut.push({ ...note, attempt: note.attempt + 1 });
          } else {
            console.log(`   ❌ [${i + 1}/${timedOutNotes.length}] Retry failed: "${note.title}" - ${errorMsg}`);
          }
        }
      }
      
      // Process successful retries into chunks and write to database
      if (retryResults.length > 0) {
        console.log(`   ✂️ Processing ${retryResults.length} successful retries into chunks...`);
        const retryChunks: ChunkData[] = [];
        
        for (const note of retryResults) {
          try {
            const plainText = htmlToPlainText(note.content || "");
            const fullText = `${note.title}\n\n${plainText}`;
            const chunks = await createChunks(fullText);
            
            chunks.forEach((chunkContent, index) => {
              retryChunks.push({
                title: note.title,
                content: plainText,
                creation_date: note.creation_date,
                modification_date: note.modification_date,
                chunk_index: index.toString(),
                total_chunks: chunks.length.toString(),
                chunk_content: chunkContent,
                cluster_id: "",
                cluster_label: "",
                cluster_confidence: "",
                cluster_summary: "",
                last_clustered: "",
              });
            });
            
            totalProcessed++;
            console.log(`     📝 "${note.title}" → ${chunks.length} chunks (retry success)`);
          } catch (error) {
            console.log(`     ❌ Failed to chunk retry "${note.title}": ${(error as Error).message}`);
            totalFailed++;
          }
        }
        
        // Write retry chunks to database
        if (retryChunks.length > 0) {
          console.log(`   💾 Writing ${retryChunks.length} retry chunks to database...`);
          
          for (let j = 0; j < retryChunks.length; j += DB_BATCH_SIZE) {
            const chunkBatch = retryChunks.slice(j, j + DB_BATCH_SIZE);
            try {
              console.log(`     🔄 Adding ${chunkBatch.length} retry chunks with embedding function...`);
              
              if ((notesTable as any)._needsManualEmbeddings) {
                // Generate embeddings manually for retry batch
                const processedBatch = await Promise.all(chunkBatch.map(async (chunk) => {
                  const embeddings = await func.computeSourceEmbeddings([chunk.chunk_content]);
                  return {
                    ...chunk,
                    // Convert to regular Array to avoid "vector.0" schema inference errors
                    // Also use Array.from() to ensure it's a plain JS array, not Float32Array
                    vector: Array.from(embeddings[0])
                  };
                }));
                
                // IMPORTANT: When manually providing vectors, we must NOT pass the embeddings option
                // or LanceDB will try to double-process or validating against broken metadata
                await notesTable.add(processedBatch);
              } else {
                await notesTable.add(chunkBatch, { 
                  embeddings: {
                    chunk_content: func
                  }
                });
              }
              
              console.log(`     ✅ Wrote ${chunkBatch.length} retry chunks to database`);
            } catch (error) {
              console.error(`     ❌ Failed to write retry chunk batch:`, error);
              console.error(`     🔍 Debug info - func:`, typeof func, 'chunkBatch length:', chunkBatch.length);
              throw error;
            }
          }
          
          totalChunks += retryChunks.length;
          const currentRowCount = await notesTable.countRows();
          console.log(`   🔍 Database now has ${currentRowCount} total rows (+${retryChunks.length} from retries)`);
        }
      }
      
      // Update the list for next retry round
      timedOutNotes.length = 0;
      timedOutNotes.push(...stillTimedOut);
      
      if (retryResults.length > 0) {
        console.log(`✅ Retry round ${retryRound + 1} complete: ${retryResults.length} notes recovered, ${stillTimedOut.length} still timing out`);
      }
    }
    
    // Final timeout summary
    if (timedOutNotes.length > 0) {
      console.log(`\n⚠️ Final timeouts: ${timedOutNotes.length} notes could not be processed even with extended timeouts`);
      timedOutNotes.forEach(note => {
        console.log(`   ⏰ "${note.title}" - failed after ${maxRetries + 1} attempts (5min → 7.5min → 10min)`);
      });
    }
  }
  
  // Final verification - check if database grew by expected amount
  const finalRowCount = await notesTable.countRows();
  const expectedFinalCount = initialRowCount + totalChunks;
  console.log(`\n🔍 Final verification: Database has ${finalRowCount} rows`);
  console.log(`📊 Expected: ${expectedFinalCount} rows (${initialRowCount} initial + ${totalChunks} new)`);
  
  if (finalRowCount !== expectedFinalCount) {
    console.error(`❌ DATABASE WRITE VERIFICATION FAILED!`);
    console.error(`   Initial rows: ${initialRowCount}`);
    console.error(`   New chunks added: ${totalChunks}`);
    console.error(`   Expected final: ${expectedFinalCount}`);
    console.error(`   Actual final: ${finalRowCount}`);
    console.error(`   Difference: ${finalRowCount - expectedFinalCount} chunks`);
    throw new Error(`Database write verification failed: ${finalRowCount}/${expectedFinalCount} total chunks (expected growth of ${totalChunks})`);
  } else {
    console.log(`✅ Database write verification successful: Added ${totalChunks} chunks, total now ${finalRowCount}`);
  }
  
  // Step 4: Save updated cache
  console.log(`\nStep 4: Updating backup cache...`);
  // Save to JSON cache as backup (database is the source of truth)
  try {
    const existingCache = await loadNotesCache();
    const existingNotes = existingCache?.notes || [];
    const mergedNotes = mergeNotesForCache(noteTitles, existingNotes);
    await saveNotesCache(mergedNotes);
  } catch (error) {
    console.log(`⚠️ Backup cache save failed (non-critical): ${(error as Error).message}`);
  }
  
  const totalTime = (performance.now() - start) / 1000;
  
  console.log(`\n✨ Complete! ${totalProcessed} notes → ${totalChunks} chunks in ${totalTime.toFixed(1)}s`);
  if (skippedCount > 0) {
    console.log(`⏩ Skipped ${skippedCount} unchanged notes (incremental mode)`);
  }
  if (totalRetries > 0) {
    console.log(`🔄 Retries: ${totalRetries} notes recovered through retry with longer timeouts`);
  }
  if (timedOutNotes.length > 0) {
    console.log(`⏰ Final timeouts: ${timedOutNotes.length} notes permanently timed out (tried 5min → 7.5min → 10min)`);
  }
  if (totalFailed > 0) {
    console.log(`❌ Failed: ${totalFailed} notes failed to process due to errors`);
  }
  
  return { 
    processed: totalProcessed, 
    totalChunks, 
    failed: totalFailed, 
    retries: totalRetries,
    finalTimeouts: timedOutNotes.length,
    skipped: skippedCount, 
    timeSeconds: totalTime 
  };
};

// Helper function to create FTS index on chunk_content
const createFTSIndex = async (notesTable: any) => {
  try {
    const indices = await notesTable.listIndices();
    if (!indices.find((index: any) => index.name === "chunk_content_idx")) {
      await notesTable.createIndex("chunk_content", {
        config: lancedb.Index.fts(),
        replace: true,
      });
      console.log(`✅ Created FTS index on chunk_content`);
    }
  } catch (error) {
    console.log(`⚠️ FTS index creation failed: ${(error as Error).message}`);
  }
};

// Replace your createNotesTable function with this smart version:
export const createNotesTableSmart = async (overrideName?: string, mode: 'fresh' | 'incremental' = 'incremental') => {
  const start = performance.now();
  const tableName = overrideName || "notes";
  
  if (mode === 'fresh') {
    try { await db.dropTable(tableName); } catch {}
    const notesTable = await db.createEmptyTable(tableName, notesTableSchema, { mode: "create", existOk: false });
    await createFTSIndex(notesTable);
    return { notesTable, existingNotes: new Map(), time: performance.now() - start };
  } 
  
  // Incremental mode
  let notesTable;
  let existingNotes = new Map();
  let needsRecovery = false;

  try {
    // Attempt 1: Open normally
    notesTable = await db.openTable(tableName);
    
    // VALIDATION: Try a lightweight read to ensure metadata is valid
    try {
      await notesTable.search("").limit(1).toArray();
      console.log(`📂 Table opened successfully and is readable`);
    } catch (readErr) {
      const msg = (readErr as Error).message;
      if (msg.includes("No embedding functions") || msg.includes("Schema") || msg.includes("vector")) {
        console.log(`⚠️ Opened table but read failed (${msg}). Triggering recovery.`);
        needsRecovery = true;
      } else {
        throw readErr; // Some other error, re-throw it
      }
    }

  } catch (error) {
    // Open failed completely (e.g. missing metadata file)
    console.log(`⚠️ Standard open failed: ${(error as Error).message}`);
    needsRecovery = true;
  }

  // RECOVERY LOGIC
  if (needsRecovery) {
    console.log(`🔧 Executing table recovery...`);
    console.log(`   The table exists with data but lacks embedding function metadata.`);
    console.log(`   We'll wrap it with the current embedding function for this session.`);
    
    try {
      // Don't try to change the schema - just wrap the existing table handle with embedding metadata
      // The table was already opened above, we just need to register the embedding function
      // Flag this table as needing manual embedding handling
      (notesTable as any)._needsManualEmbeddings = true;
      console.log(`✅ Recovery mode enabled. Will use manual embeddings for new data.`);
    } catch (recErr) {
      console.error(`\n❌ Recovery setup failed: ${(recErr as Error).message}`);
      throw new Error(`Recovery failed: ${(recErr as Error).message}. You may need to use --mode=fresh`);
    }
  }

  // Load existing notes for deduplication
  try {
    const existing = await notesTable.search("").limit(50000).toArray();
    existing.forEach(note => {
      if (note.title) {
        existingNotes.set(note.title, {
          modification_date: note.modification_date,
          row: note
        });
      }
    });
    console.log(`📊 Found ${existingNotes.size} existing notes for comparison`);
  } catch (err) {
    console.log(`⚠️ Error reading existing data (non-fatal): ${(err as Error).message}`);
  }
  
  await createFTSIndex(notesTable);
  return { notesTable, existingNotes, time: performance.now() - start };
};

export const createNotesTable = async (overrideName?: string) => {
  // Use the smart version with incremental mode by default
  return await createNotesTableSmart(overrideName, 'incremental');
};

// Handle tool execution
server.setRequestHandler(CallToolRequestSchema, async (request, c) => {
  const { notesTable } = await createNotesTable();
  const { name, arguments: args } = request.params;

  try {
    if (name === "create-note") {
      // Remove createNote functionality since it's not needed
      return createTextResponse(`Create note functionality not implemented.`);
    } else if (name === "list-notes") {
      const totalChunks = await notesTable.countRows();
      // Get unique note titles to count actual notes
      const allChunks = await notesTable.search("").limit(50000).toArray();
      const uniqueNotes = new Set(allChunks.map(chunk => chunk.title));
      return createTextResponse(
        `There are ${uniqueNotes.size} notes (${totalChunks} chunks) in your Apple Notes database.`
      );
    } else if (name == "get-note") {
      try {
        const { title, creation_date } = GetNoteSchema.parse(args);
        const note = await getNoteDetailsByTitle(title, creation_date);

        return createTextResponse(`${JSON.stringify(note, null, 2)}`);
      } catch (error) {
        return createTextResponse((error as Error).message);
      }
    } else if (name === "index-notes") {
      // Use the enhanced method by default for better reliability
      const { mode } = IndexNotesSchema.parse(args);
      const { processed, totalChunks, failed, skipped, timeSeconds } = await fetchAndIndexAllNotes(notesTable, undefined, mode);
      
      let message = `Successfully indexed ${processed} notes into ${totalChunks} chunks in ${timeSeconds.toFixed(1)}s using enhanced method.\n\n` +
        `📊 Summary:\n` +
        `• Notes processed: ${processed}\n` +
        `• Chunks created: ${totalChunks}\n` +
        `• Failed: ${failed}\n`;
      
      if (skipped > 0) {
        message += `• Skipped unchanged: ${skipped}\n`;
      }
      
      message += `• Average chunks per note: ${processed > 0 ? (totalChunks/processed).toFixed(1) : '0'}\n` +
        `• Processing time: ${timeSeconds.toFixed(1)} seconds\n` +
        `• Mode: ${mode}\n\n` +
        `✨ Enhanced indexing handles duplicate titles better by using creation dates!\n`;
      
      if (mode === 'incremental' && skipped > 0) {
        message += `⚡ Incremental mode: Only processed new/modified notes. ${skipped} notes unchanged.\n`;
      }
      
      message += `Your notes are now ready for semantic search using the "search-notes" tool!`;
      
      return createTextResponse(message);
    } else if (name === "index-notes-enhanced") {
      const { processed, totalChunks, failed, timeSeconds } = await fetchAndIndexAllNotes(notesTable);
      return createTextResponse(
        `Successfully indexed ${processed} notes into ${totalChunks} chunks in ${timeSeconds.toFixed(1)}s using enhanced method.\n\n` +
        `📊 Summary:\n` +
        `• Notes processed: ${processed}\n` +
        `• Chunks created: ${totalChunks}\n` +
        `• Failed: ${failed}\n` +
        `• Average chunks per note: ${(totalChunks/processed).toFixed(1)}\n` +
        `• Processing time: ${timeSeconds.toFixed(1)} seconds\n\n` +
        `✨ Enhanced indexing handles duplicate titles better by using creation dates!\n` +
        `Your notes are now ready for semantic search using the "search-notes" tool!`
      );
    } else if (name === "search-notes") {
      const { query } = QueryNotesSchema.parse(args);
      const combinedResults = await searchAndCombineResults(notesTable, query);
      return createTextResponse(JSON.stringify(combinedResults, null, 2));
    } else {
      throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new Error(
        `Invalid arguments: ${error.errors
          .map((e) => `${e.path.join(".")}: ${e.message}`)
          .join(", ")}`
      );
    }
    throw error;
  }
});

// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Local Machine MCP Server running on stdio");

const createTextResponse = (text: string) => ({
  content: [{ type: "text", text }],
});

/**
 * Enhanced search relying purely on semantic content analysis
 */
export const searchAndCombineResults = async (
  notesTable: lancedb.Table,
  query: string,
  displayLimit = 5,
  minCosineSimilarity = 0.05
) => {
  console.log(`🔍 Semantic search for: "${query}"`);
  console.log(`📊 Table has ${await notesTable.countRows()} chunks`);
  
  const noteResults = new Map(); // title -> best result for that note
  
  // Strategy 1: Vector search on chunks
  console.log(`\n1️⃣ Vector semantic search on chunks...`);
  try {
    const vectorResults = await notesTable.search(query, "vector").toArray();
    
    if (vectorResults.length > 0) {
      console.log(`🎯 Found ${vectorResults.length} relevant chunks`);
      
      vectorResults.forEach(chunk => {
        const distance = chunk._distance || 0;
        const cosineSimilarity = Math.max(0, 1 - (distance * distance / 2));
        
        if (cosineSimilarity > minCosineSimilarity) {
          const existing = noteResults.get(chunk.title);
          
          if (!existing || cosineSimilarity > existing._relevance_score) {
            noteResults.set(chunk.title, {
              title: chunk.title,
              content: chunk.content,
              creation_date: chunk.creation_date,
              modification_date: chunk.modification_date,
              _relevance_score: cosineSimilarity * 100,
              _source: 'vector_semantic',
              _best_chunk_index: chunk.chunk_index,
              _total_chunks: chunk.total_chunks,
              _matching_chunk_content: chunk.chunk_content
            });
          }
        }
      });
      
      console.log(`📋 Unique notes from vector search: ${noteResults.size}`);
    }
  } catch (error) {
    console.log(`❌ Vector Error: ${(error as Error).message}`);
  }
  
  // Strategy 2: FTS search on chunk content
  console.log(`\n2️⃣ Full-text search on chunks...`);
  try {
    const ftsResults = await notesTable.search(query, "fts", "chunk_content").toArray();

    // Compute query embedding once for all FTS results
    let queryEmbedding: number[] | null = null;
    try {
      queryEmbedding = await func.computeQueryEmbeddings(query);
    } catch (e) {
      console.log(`⚠️ Could not compute query embedding for FTS scoring: ${(e as Error).message}`);
    }

    // Helper to compute cosine similarity
    const cosineSimilarity = (a: number[], b: number[]) => {
      if (!a || !b || a.length !== b.length) return 0;
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      if (normA === 0 || normB === 0) return 0;
      return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    };

    ftsResults.forEach(chunk => {
      if (!noteResults.has(chunk.title)) {
        let score = 70; // fallback
        if (queryEmbedding && Array.isArray(chunk.vector) && chunk.vector.length === queryEmbedding.length) {
          score = Math.max(0, cosineSimilarity(queryEmbedding, chunk.vector)) * 100;
        }
        noteResults.set(chunk.title, {
          title: chunk.title,
          content: chunk.content,
          creation_date: chunk.creation_date,
          modification_date: chunk.modification_date,
          _relevance_score: score,
          _source: 'fts',
          _best_chunk_index: chunk.chunk_index,
          _total_chunks: chunk.total_chunks,
          _matching_chunk_content: chunk.chunk_content
        });
      }
    });

    console.log(`📝 FTS results: ${ftsResults.length} chunks`);
  } catch (error) {
    console.log(`❌ FTS Error: ${(error as Error).message}`);
  }
  
  // Strategy 3: Database-level exact phrase matching (much more efficient)
  console.log(`\n3️⃣ Database-level exact phrase search...`);
  try {
    // Use SQL-like filtering instead of loading all chunks
    const queryWords = query.toLowerCase().split(/\s+/).filter(word => word.length > 2);
    
    if (queryWords.length > 0) {
      // Search for chunks that contain all query words
      const sqlFilter = `LOWER(chunk_content) LIKE '%${queryWords.join("%' AND LOWER(chunk_content) LIKE '%")}%'`;
      
      const exactMatches = await notesTable
        .search("")
        .where(sqlFilter)
        .limit(100)
        .toArray();
      
      console.log(`📋 Database exact matches: ${exactMatches.length} chunks`);
      
      exactMatches.forEach(chunk => {
        if (!noteResults.has(chunk.title)) {
          // Check if it's a real exact match (for better scoring)
          const isExactMatch = chunk.chunk_content?.toLowerCase().includes(query.toLowerCase()) ||
                              chunk.title?.toLowerCase().includes(query.toLowerCase());
          
          noteResults.set(chunk.title, {
            title: chunk.title,
            content: chunk.content,
            creation_date: chunk.creation_date,
            modification_date: chunk.modification_date,
            _relevance_score: isExactMatch ? 100 : 85,
            _source: isExactMatch ? 'exact_match' : 'partial_match',
            _best_chunk_index: chunk.chunk_index,
            _total_chunks: chunk.total_chunks,
            _matching_chunk_content: chunk.chunk_content
          });
        }
      });
    }
  } catch (error) {
    console.log(`❌ Database search error: ${(error as Error).message}`);
    // Fallback: try a simpler approach
    console.log(`🔄 Trying fallback search...`);
    try {
      const fallbackResults = await notesTable
        .search("")
        .limit(1000) // Much smaller limit
        .toArray();
      
      const queryRegex = new RegExp(`\\b${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      
      const matches = fallbackResults.filter(chunk => {
        const titleMatch = queryRegex.test(chunk.title || '');
        const contentMatch = queryRegex.test(chunk.chunk_content || '');
        return titleMatch || contentMatch;
      });
      
      console.log(`📋 Fallback matches: ${matches.length} chunks`);
      
      matches.forEach(chunk => {
        if (!noteResults.has(chunk.title)) {
          noteResults.set(chunk.title, {
            title: chunk.title,
            content: chunk.content,
            creation_date: chunk.creation_date,
            modification_date: chunk.modification_date,
            _relevance_score: 90,
            _source: 'fallback_exact',
            _best_chunk_index: chunk.chunk_index,
            _total_chunks: chunk.total_chunks,
            _matching_chunk_content: chunk.chunk_content
          });
        }
      });
    } catch (fallbackError) {
      console.log(`❌ Fallback also failed: ${(fallbackError as Error).message}`);
    }
  }
  
  // Combine and rank results
  const combinedResults = Array.from(noteResults.values())
    .sort((a, b) => b._relevance_score - a._relevance_score);

  console.log(`\n📊 Final results: ${combinedResults.length} notes (from ${noteResults.size} total matches)`);

  if (combinedResults.length > 0) {
    combinedResults.forEach((result, idx) => {
      console.log(`  ${idx + 1}. "${result.title}" (score: ${result._relevance_score.toFixed(1)}, source: ${result._source}, chunk: ${result._best_chunk_index}/${result._total_chunks})`);
    });
  }

  return combinedResults.map(result => ({
    title: result.title,
    creation_date: result.creation_date,
    modification_date: result.modification_date,
    _relevance_score: result._relevance_score,
    _source: result._source,
    _best_chunk_index: result._best_chunk_index,
    _total_chunks: result._total_chunks,
    _matching_chunk_preview: result._matching_chunk_content
  }));
};