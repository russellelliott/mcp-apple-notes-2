#!/usr/bin/env node
/**
 * Quick Apple Notes Window Checker
 * 
 * Simply outputs the titles of all open Notes windows.
 * No background tracking, no database - just instant results.
 * 
 * Usage: node check-notes.js
 */

import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

async function main() {
  console.log("🔍 Checking open Notes windows...\n");
  
  try {
    const script = `
      (() => {
        const Notes = Application("Notes");
        if (!Notes.running()) return [];
        return Notes.windows.name().filter(name => name && name !== "Notes");
      })();
     `;
    
    const result = await execAsync(`osascript -l JavaScript ${script}`);
    
    // Parse the output
    let titles: string[];
    try {
      titles = JSON.parse(result.stdout.trim());
    } catch {
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
    
    if (titles.length === 0) {
      console.log("ℹ️  No Notes windows are currently open");
     } else {
      console.log(`✅ Found ${titles.length} open Note(s):\n`);
      titles.forEach((title, index) => {
        console.log(`   ${index + 1}. ${title}`);
       });
     }
    
   } catch (error) {
    console.error(`❌ Error: ${(error as Error).message}`);
    process.exit(1);
   }
}

main();
