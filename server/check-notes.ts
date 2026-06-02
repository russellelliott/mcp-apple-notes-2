#!/usr/bin/env bun

import { writeFileSync, unlinkSync } from "fs";
import { execSync } from "child_process";
import { tmpdir } from "os";
import { join } from "path";

function main() {
  console.log("🔍 Checking open Notes windows...\n");

  try {
    const tempFile = join(tmpdir(), `notes-check-${Date.now()}.js`);

    const scriptContent = `
(() => {
  const Notes = Application("Notes");

  if (!Notes.running()) {
    return JSON.stringify([]);
  }

  const windows = Notes.windows.name();
  return JSON.stringify(
    windows.filter(name => name && name !== "Notes")
  );
})();
`;

    writeFileSync(tempFile, scriptContent);

    const output = execSync(`osascript -l JavaScript "${tempFile}"`, {
      encoding: "utf-8",
      timeout: 10000,
    });

    try {
      unlinkSync(tempFile);
    } catch {}

    const titles: string[] = JSON.parse(output.trim());

    if (titles.length === 0) {
      console.log("ℹ️  No Notes windows are currently open");
    } else {
      console.log(`✅ Found ${titles.length} open Note(s):\n`);
      titles.forEach((title, index) => {
        console.log(`  ${index + 1}. ${title}`);
      });
    }
  } catch (error: any) {
    if (error.status === 124) {
      console.log("⏰ JXA execution timed out after 10 seconds");
      process.exit(1);
    } else {
      console.error(`❌ Error: ${error.message}`);
      process.exit(1);
    }
  }
}

main();