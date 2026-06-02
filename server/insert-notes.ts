#!/usr/bin/env bun
/**
 * Insert Open Notes into notes_interactions Table
 *
 * Usage:
 *   bun server/insert-notes.ts
 */

import { writeFileSync, unlinkSync } from "fs";
import { execSync } from "child_process";
import { tmpdir } from "os";
import { join } from "path";
import * as lancedb from "@lancedb/lancedb";
import { LanceSchema } from "@lancedb/lancedb/embedding";
import { Utf8 } from "apache-arrow";

type NoteInteractionRow = {
  title: string;
  last_opened: string;
  interaction_log: string;
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function makeNotesJxaScript(): string {
  return `
(() => {
  const Notes = Application("Notes");

  if (!Notes.running()) {
    return JSON.stringify([]);
  }

  const windows = Notes.windows.name();
  const titles = windows.filter(name => name && name !== "Notes");

  return JSON.stringify(titles);
})();
`;
}

async function getOpenNoteTitles(): Promise<string[]> {
  const tempFile = join(tmpdir(), `notes-check-${Date.now()}.js`);

  try {
    writeFileSync(tempFile, makeNotesJxaScript());

    const output = execSync(`osascript -l JavaScript "${tempFile}"`, {
      encoding: "utf-8",
      timeout: 10000,
    });

    const parsed = JSON.parse(output.trim());

    if (!Array.isArray(parsed)) {
      throw new Error("JXA output was not an array");
    }

    return parsed.filter(
      (value): value is string =>
        typeof value === "string" && value.trim().length > 0
    );
  } finally {
    try {
      unlinkSync(tempFile);
    } catch {}
  }
}

async function getOrCreateInteractionsTable(db: any) {
  const tableNames = await db.tableNames();

  if (!tableNames.includes("notes_interactions")) {
    console.log("   Creating notes_interactions table...");

    await db.createTable("notes_interactions", [], {
      schema: LanceSchema({
        title: new Utf8(),
        last_opened: new Utf8(),
        interaction_log: new Utf8(),
      }),
    });

    console.log("   ✅ notes_interactions table created");
  }

  return db.openTable("notes_interactions");
}

function buildInteractionLog(existingLog: string | null | undefined, now: string) {
  let parsed: Array<{ dt: string; type: string }> = [];

  try {
    const maybeParsed = JSON.parse(existingLog || "[]");
    if (Array.isArray(maybeParsed)) {
      parsed = maybeParsed;
    }
  } catch {}

  parsed.push({ dt: now, type: "opened" });
  return JSON.stringify(parsed);
}

async function upsertOpenNotes(table: any, titles: string[]) {
  console.log("\n📝 Inserting notes into notes_interactions...\n");

  for (const title of titles) {
    const now = new Date().toISOString();
    const escapedTitle = escapeSqlString(title);

    try {
      const existing = await table
        .query()
        .where(`title = '${escapedTitle}'`)
        .limit(1)
        .toArray();

      if (existing.length > 0) {
        const record = existing[0] as NoteInteractionRow;
        const interaction_log = buildInteractionLog(record.interaction_log, now);

        await table.update(
          {
            last_opened: now,
            interaction_log,
          },
          {
            where: `title = '${escapedTitle}'`,
          }
        );

        console.log(`   ✏️ Updated: ${title}`);
      } else {
        await table.add([
          {
            title,
            last_opened: now,
            interaction_log: JSON.stringify([{ dt: now, type: "opened" }]),
          },
        ]);

        console.log(`   ➕ Created: ${title}`);
      }
    } catch (error: any) {
      console.log(`   ❌ Error for "${title}": ${error.message}`);
    }
  }
}

async function main() {
  console.log("🔍 Getting open Notes windows...\n");

  let db: any;

  try {
    const titles = await getOpenNoteTitles();

    if (titles.length === 0) {
      console.log("ℹ️  No Notes windows are currently open");
      process.exit(0);
    }

    console.log(`✅ Found ${titles.length} open Note(s):\n`);
    titles.forEach((title, index) => {
      console.log(`   ${index + 1}. ${title}`);
    });

    console.log("\n📊 Connecting to notes_interactions table...");

    const dbPath = join(process.env.HOME || "", ".mcp-apple-notes", "data");
    db = await lancedb.connect(dbPath);

    const table = await getOrCreateInteractionsTable(db);

    await upsertOpenNotes(table, titles);

    const count = await table.countRows();
    console.log("\n✅ Done! All open notes processed.");
    console.log(`📊 Total rows in notes_interactions: ${count}`);
  } catch (error: any) {
    if (error?.status === 124) {
      console.error("⏰ JXA execution timed out after 10 seconds");
      console.error("Make sure Apple Notes is running and automation permissions are allowed");
    } else {
      console.error(`❌ Error: ${error?.message || error}`);
    }
    process.exit(1);
  } finally {
    try {
      await db?.close?.();
    } catch {}
  }
}

main();