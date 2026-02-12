#!/usr/bin/env bun
import { createNotesTableSmart, fetchAndIndexAllNotes } from "./index.js";
import * as readline from 'readline';

function askQuestion(query: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise(resolve => {
    rl.question(query, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

async function main() {
  console.log("🚀 Enhanced Apple Notes Indexing\n");
  
  // Parse command line arguments
  const args = process.argv.slice(2);
  const maxNotesArg = args.find(arg => arg.startsWith('--max='));
  const modeArg = args.find(arg => arg.startsWith('--mode='));
  const tableArg = args.find(arg => arg.startsWith('--table='));
  
  const maxNotes = maxNotesArg ? parseInt(maxNotesArg.split('=')[1]) : undefined;
  const mode = (modeArg?.split('=')[1] as 'fresh' | 'incremental' | 'incremental-since') || 'incremental'; // Default to incremental
  const tableName = tableArg?.split('=')[1] || 'notes'; // Default to 'notes'
  
  // Fresh mode confirmation
  if (mode === 'fresh') {
    console.log(`⚠️  FRESH MODE WARNING: This will completely reset the database and reprocess all notes.`);
    console.log(`📝 To confirm, please type "reset database" (without quotes):`);
    
    const confirmation = await askQuestion('> ');
    
    if (confirmation.trim() !== 'reset database') {
      console.log(`❌ Confirmation failed. Exiting without making changes.`);
      process.exit(0);
    }
    
    console.log(`✅ Fresh mode confirmed. Proceeding with database reset...\n`);
  }
  
  const modeDescriptions = {
    'fresh': 'Fresh rebuild',
    'incremental': 'Incremental updates',
    'incremental-since': 'Incremental updates (Date-based)'
  };

  console.log(`📊 Mode: ${modeDescriptions[mode]}`);
  console.log(`🔧 Method: Enhanced (title + creation date) - handles duplicate titles better`);
  console.log(`📁 Table: ${tableName}`);
  if (maxNotes) {
    console.log(`🎯 Limit: ${maxNotes} notes`);
  }
  
  try {
    console.log("📁 Setting up notes database...");
    const { notesTable } = await createNotesTableSmart(tableName, mode);
    console.log(`✅ Database setup complete`);
    
    console.log("\n📝 Starting enhanced indexing...");
    
    // Use the enhanced method that fetches by title and creation date with mode support
    const result = await fetchAndIndexAllNotes(notesTable, maxNotes, mode);
    
    console.log("\n=== Enhanced Indexing Complete ===");
    console.log(`📊 Stats:`);
    console.log(`• Notes processed: ${result.processed}`);
    console.log(`• Chunks created: ${result.totalChunks}`);
    console.log(`• Failed: ${result.failed} notes`);
    if (result.skipped > 0) {
      console.log(`• Skipped unchanged: ${result.skipped} notes`);
    }
    console.log(`• Time taken: ${result.timeSeconds.toFixed(2)} seconds`);
    console.log(`• Mode: ${mode}`);
    
    console.log("\n✨ Notes are now ready for semantic search!");
    console.log("🎯 Enhanced method handles duplicate note titles by using creation dates for precise fetching.");
    
    if (mode === 'incremental' && result.skipped > 0) {
      console.log(`⚡ Incremental mode: Only processed new/modified notes. Cache saved for future runs.`);
    }
    
    process.exit(0);
  } catch (error) {
    console.error("\n❌ Error:", error);
    process.exit(1);
  }
}

main();