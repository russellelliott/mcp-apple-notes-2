#!/usr/bin/env bun
/**
 * Insert Open Notes into notes_interactions Table
 *
 * Behavior:
 * - Reads currently open Apple Notes window titles via JXA
 * - Uses note title as the logical key / bucket
 * - If multiple notes have the same title, they collapse into one row
 * - Replaces existing matching rows using delete + add
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

type NoteInteraction = {
  dt: string;
  type: "opened";
};

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

  return await db.openTable("notes_interactions");
}

function parseInteractionLog(existingLog: string | null | undefined): NoteInteraction[] {
  try {
    const parsed = JSON.parse(existingLog || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) =>
        item &&
        typeof item === "object" &&
        typeof item.dt === "string" &&
        typeof item.type === "string"
    ) as NoteInteraction[];
  } catch {
    return [];
  }
}

async function upsertOpenNotes(table: any, titles: string[]) {
  console.log("\n📝 Inserting notes into notes_interactions...\n");

  let created = 0;
  let updated = 0;
  let failed = 0;

  const uniqueTitles = [...new Set(titles)];

  for (const title of uniqueTitles) {
    const now = new Date().toISOString();
    const escapedTitle = escapeSqlString(title);
    const whereClause = `title = '${escapedTitle}'`;

    try {
      const existingRows = await table
        .query()
        .where(whereClause)
        .toArray();

      let mergedLog: NoteInteraction[] = [];

      for (const row of existingRows as NoteInteractionRow[]) {
        mergedLog.push(...parseInteractionLog(row.interaction_log));
      }

      mergedLog.push({ dt: now, type: "opened" });

      if (existingRows.length > 0) {
        await table.delete(whereClause);
        await table.add([
          {
            title,
            last_opened: now,
            interaction_log: JSON.stringify(mergedLog),
          },
        ]);

        console.log(`   ✏️ Updated bucket: ${title} (${existingRows.length} prior row(s))`);
        updated += 1;
      } else {
        await table.add([
          {
            title,
            last_opened: now,
            interaction_log: JSON.stringify([{ dt: now, type: "opened" }]),
          },
        ]);

        console.log(`   ➕ Created bucket: ${title}`);
        created += 1;
      }
    } catch (error: any) {
      console.log(`   ❌ Error for "${title}": ${error.message}`);
      failed += 1;
    }
  }

  return { created, updated, failed, uniqueCount: uniqueTitles.length };
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

    const result = await upsertOpenNotes(table, titles);
    const count = await table.countRows();

    console.log("\n📈 Results:");
    console.log(`   Open windows seen: ${titles.length}`);
    console.log(`   Unique title buckets: ${result.uniqueCount}`);
    console.log(`   Created: ${result.created}`);
    console.log(`   Updated: ${result.updated}`);
    console.log(`   Failed: ${result.failed}`);
    console.log(`   Total rows in notes_interactions: ${count}`);

    if (result.failed === 0) {
      console.log("\n✅ Done! All open notes processed successfully.");
    } else {
      console.log("\n⚠️ Done, but some notes failed to process.");
      process.exitCode = 1;
    }
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