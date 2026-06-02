#!/usr/bin/env node
/**
 * Quick Apple Notes Window Checker
 * 
 * Simply outputs the titles of all open Notes windows.
 * Uses a temporary script file to avoid shell escaping issues.
 * 
 * Usage: node quick-notes.js
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";

function main() {
  console.log("🔍 Checking open Notes windows...\n");
  
  try {
    // Create a temporary JXA script file
    const tempScript = path.join(os.tmpdir(), `notes-check-${Date.now()}.js`);
    const scriptContent = `
      const Notes = Application("Notes");
      if (!Notes.running()) {
        print("NOTES_NOT_RUNNING");
        quit();
      }
      const windows = Notes.windows.name();
      print("WINDOWS_COUNT:" + windows.length);
      for (let i = 0; i < windows.length; i++) {
        const name = windows[i];
        if (name && name !== "Notes") {
          print("WINDOW:" + name);
        }
      }
      quit();
    `;
    
    fs.writeFileSync(tempScript, scriptContent);
    
    // Execute the script with a timeout
    const timeout = 10000; // 10 second timeout
    const startTime = Date.now();
    
    let output = "";
    let timedOut = false;
    
    try {
      output = execSync(`osascript "${tempScript}"`, { 
        timeout: timeout,
        encoding: "utf-8"
      });
    } catch (error) {
      if (error.status === 124) {
        timedOut = true;
        output = error.stdout || "";
      } else {
        throw error;
      }
    }
    
    // Clean up temp file
    try {
      fs.unlinkSync(tempScript);
    } catch {}
    
    const elapsed = Date.now() - startTime;
    
    if (timedOut) {
      console.log("⏰ JXA execution timed out after 10 seconds");
      console.log("Make sure Apple Notes is running and has automation permissions");
      process.exit(1);
    }
    
    // Parse the output
    const lines = output.split("\n").filter(line => line.trim().length > 0);
    
    // Check if Notes is not running
    if (lines.some(line => line.includes("NOTES_NOT_RUNNING"))) {
      console.log("ℹ️  Apple Notes is not running");
      process.exit(0);
    }
    
    // Get window count
    const countLine = lines.find(line => line.includes("WINDOWS_COUNT:"));
    if (countLine) {
      const count = parseInt(countLine.split(":")[1], 10);
      
      if (isNaN(count) || count === 0) {
        console.log("ℹ️  No Notes windows are currently open");
        } else {
        console.log(`✅ Found ${count} open Note(s):\n`);
        
        const windowLines = lines.filter(line => line.includes("WINDOW:"));
        windowLines.forEach((line, index) => {
          const title = line.replace("WINDOW:", "");
          console.log(`    ${index + 1}. ${title}`);
          });
        }
      }
    
    } catch (error) {
    console.error(`❌ Error: ${(error as Error).message}`);
    process.exit(1);
    }
}

main();
