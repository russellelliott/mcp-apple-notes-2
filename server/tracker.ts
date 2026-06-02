#!/usr/bin/env node
/**
 * Apple Notes Live Tracker
 * 
 * A standalone background process that tracks note interactions
 * (opens and modifications) in a separate LanceDB table.
 * Runs every 10 seconds to poll open note windows.
 * 
 * Usage: node tracker.ts
 */

import * as lancedb from "@lancedb/lancedb";
import { exec } from "child_process";
import { promisify } from "util";
import path from "node:path";
import os from "node:os";
import { type Float, Utf8 } from "apache-arrow";
import { LanceSchema } from "@lancedb/lancedb/embedding";

const execAsync = promisify(exec);

// Configuration
const DATA_DIR = path.join(os.homedir(), ".mcp-apple-notes");
const INTERACTIONS_DB_PATH = path.join(DATA_DIR, "data");
const INTERACTIONS_TABLE_NAME = "notes_interactions";
const TRACKING_INTERVAL_MS = 10000; // 10 seconds

/**
 * Execute JXA script to get open note window titles
 */
async function getOpenNoteTitles(): Promise<string[]> {
  try {
    console.log("  📝 Running JXA script to get open notes...");
    
    const script = `
      (() => {
        const Notes = Application("Notes");
        if (!Notes.running()) {
          print("Notes app is not running");
          return [];
        }
        const windows = Notes.windows.name();
        print("Found " + windows.length + " windows");
        return windows.filter(name => name && name !== "Notes");
      })();
    `;
    
    console.log("  🚀 Executing JXA...");
    const result = await execAsync(`osascript -l JavaScript ${script}`);
    console.log(`  📤 JXA output: ${result.stdout}`);
    
     // Parse the output - it might be JSON or a direct array representation
    let titles: string[];
    try {
      titles = JSON.parse(result.stdout.trim());
    } catch {
       // If JSON parsing fails, try to extract array from output
      const match = result.stdout.trim().match(/\[([\s\S]*)\]/);
      if (match) {
        titles = match[1]
           .split(',')
           .map(s => s.trim().replace(/^"|"$/g, ''))
           .filter(s => s.length > 0);
       } else {
        titles = [];
       }
     }
    
    console.log(`  ✅ Found ${titles.length} open notes:`, titles);
    return Array.isArray(titles) ? titles : [];
   } catch (error) {
    console.log(`  ❌ JXA execution failed: ${(error as Error).message}`);
    return [];
   }
}

/**
 * Initialize the interactions table if it doesn't exist
 */
async function initInteractionsTable(): Promise<lancedb.Table> {
  const db = await lancedb.connect(INTERACTIONS_DB_PATH);
  
  // Check if table exists
  const tableNames = await db.tableNames();
  if (!tableNames.includes(INTERACTIONS_TABLE_NAME)) {
    console.log("📊 Creating notes_interactions table...");
    
    // Create empty table with proper Arrow schema
    await db.createTable(INTERACTIONS_TABLE_NAME, [], {
      schema: LanceSchema({
        title: new Utf8(),
        last_opened: new Utf8(),
        interaction_log: new Utf8() // JSON string stored as text
      })
    });
    
    console.log("✅ notes_interactions table created");
  }
  
  return await db.openTable(INTERACTIONS_TABLE_NAME);
}

/**
 * Upsert a note interaction record
 * @param title - Note title (primary key)
 * @param eventType - "opened" or "modified"
 */
async function upsertInteraction(
  table: lancedb.Table,
  title: string,
  eventType: "opened" | "modified"
): Promise<void> {
  const now = new Date().toISOString();
  const event = { dt: now, type: eventType };
  
  try {
    // Check if title exists
    const existing = await table
      .query()
      .where(`title = '${title.replace(/'/g, "''")}'`)
      .limit(1)
      .toArray();
  
    if (existing.length > 0) {
      // Update existing record
      const record = existing[0];
      const log = JSON.parse(record.interaction_log || "[]");
      log.push(event);
      
      await table.update(
        { last_opened: now, interaction_log: JSON.stringify(log) },
        { where: `title = '${title.replace(/'/g, "''")}'` }
      );
      
      console.log(`   ✏️ Updated interaction for "${title}" (${eventType})`);
    } else {
      // Insert new record
      await table.add([{
        title,
        last_opened: now,
        interaction_log: JSON.stringify([event])
      }]);
      
      console.log(`   ➕ Created new interaction record for "${title}" (${eventType})`);
    }
  } catch (error) {
    console.error(`   ❌ Error upserting interaction for "${title}": ${(error as Error).message}`);
  }
}

/**
 * Process all open notes
 */
async function processOpenNotes(table: lancedb.Table): Promise<void> {
  console.log(`\n🔍 Checking open notes...`);
  
  const titles = await getOpenNoteTitles();
  
  if (titles.length === 0) {
    console.log("  ℹ️ No Notes windows are currently open");
    return;
  }
  
  console.log(`   📝 Found ${titles.length} open note(s):`);
  titles.forEach(title => console.log(`      • ${title}`));
  
  // Upsert each open note
  for (const title of titles) {
    await upsertInteraction(table, title, "opened");
  }
}

/**
 * Main tracking loop
 */
async function startTracker(): Promise<void> {
  console.log("🚀 Apple Notes Live Tracker Started");
  console.log(`📁 Database: ${INTERACTIONS_DB_PATH}`);
  console.log(`⏱️  Check interval: ${TRACKING_INTERVAL_MS / 1000}s`);
  console.log("Press Ctrl+C to stop\n");
  
  // Initialize table
  const table = await initInteractionsTable();
  
  // Initial check
  await processOpenNotes(table);
  
  // Start periodic tracking
  const intervalId = setInterval(async () => {
    try {
      await processOpenNotes(table);
    } catch (error) {
      console.error(`   ❌ Error in tracking cycle: ${(error as Error).message}`);
    }
  }, TRACKING_INTERVAL_MS);
  
  // Handle graceful shutdown
  process.on("SIGINT", async () => {
    console.log("\n⏹️  Tracker stopping...");
    clearInterval(intervalId);
    process.exit(0);
  });
  
  process.on("SIGTERM", async () => {
    console.log("\n⏹️  Tracker stopping...");
    clearInterval(intervalId);
    process.exit(0);
  });
}

// Start the tracker
startTracker().catch(error => {
  console.error("❌ Tracker failed to start:", error);
  process.exit(1);
});
