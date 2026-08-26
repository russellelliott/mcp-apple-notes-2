#!/usr/bin/env python3
"""Prune obsolete LanceDB Apple Notes tables.

Keeps only:
  - notes
  - notes_interactions
    - the 3 newest notes_backup_<timestamp> tables

Drops:
  - older notes_backup_<timestamp> tables
  - notes_new_copy
  - notes_broken_backup
  - every table whose name starts with test-notes-

Default database directory:
  $HOME/.mcp-apple-notes/data

Usage:
  python prune_lancedb_backups.py --dry-run
  python prune_lancedb_backups.py --apply
  python prune_lancedb_backups.py --uri /other/lancedb/directory --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import lancedb
except ImportError:
    sys.exit("Missing dependency: install it in this environment with `pip install lancedb`.")

BACKUP_RE = re.compile(r"^notes_backup_(\d+)$")
PROTECTED_TABLES = {"notes", "notes_interactions"}
EXPLICITLY_DISPOSABLE = {"notes_new_copy", "notes_broken_backup"}
TEST_PREFIX = "test-notes-"
DEFAULT_URI = Path.home() / ".mcp-apple-notes" / "data"


def discover_table_names(db: "lancedb.DBConnection", uri: Path) -> list[str]:
    """Return table names using the same source as list_tables.py (filesystem)."""
    registered = set(db.table_names())
    physical = {p.stem for p in uri.glob("*.lance") if not p.stem.startswith(".")}

    only_registered = sorted(registered - physical)
    only_physical = sorted(physical - registered)
    if only_registered or only_physical:
        print("WARNING: LanceDB metadata and physical table folders differ.")
        if only_registered:
            print(f"  Metadata-only tables: {', '.join(only_registered)}")
        if only_physical:
            print(f"  Folder-only tables: {', '.join(only_physical)}")

    return sorted(registered | physical)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep protected tables and the 10 newest timestamped notes backups."
    )
    parser.add_argument(
        "--uri",
        default=str(DEFAULT_URI),
        help=f"LanceDB database directory (default: {DEFAULT_URI}).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print planned deletions only.")
    mode.add_argument("--apply", action="store_true", help="Actually drop the planned tables.")
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of newest notes_backup_<timestamp> tables to keep (default: 3).",
    )
    args = parser.parse_args()

    if args.keep < 0:
        parser.error("--keep must be zero or greater")

    uri = Path(args.uri).expanduser()
    if not uri.exists():
        parser.error(f"Database path does not exist: {uri}")

    db = lancedb.connect(str(uri))
    table_names = discover_table_names(db, uri)

    backups: list[tuple[int, str]] = []
    to_drop: set[str] = set()
    untouched: list[str] = []

    for name in table_names:
        backup_match = BACKUP_RE.fullmatch(name)
        if backup_match:
            backups.append((int(backup_match.group(1)), name))
        elif name in EXPLICITLY_DISPOSABLE or name.startswith(TEST_PREFIX):
            to_drop.add(name)
        elif name not in PROTECTED_TABLES:
            untouched.append(name)

    # Match run_list_tables ordering: newest timestamp first.
    backups.sort(key=lambda item: item[0], reverse=True)
    retained_backups = backups[: args.keep]
    to_drop.update(name for _, name in backups[args.keep:])

    print(f"LanceDB URI: {uri}")
    print(f"Tables found: {len(table_names)}")
    print("Protected tables:")
    for name in sorted(PROTECTED_TABLES):
        print(f"  KEEP    {name} ({'present' if name in table_names else 'NOT FOUND'})")

    print(f"Newest timestamped backups retained: {len(retained_backups)}")
    for _, name in retained_backups:
        print(f"  KEEP    {name}")

    print(f"Tables to drop: {len(to_drop)}")
    action = "WOULD DROP" if args.dry_run else "DROP"
    for name in sorted(to_drop):
        print(f"  {action}    {name}")

    if untouched:
        print("Other tables left untouched:")
        for name in untouched:
            print(f"  SKIP    {name}")

    if args.dry_run:
        print("\nDry run complete. No tables were deleted.")
        return 0

    for name in sorted(to_drop):
        db.drop_table(name)

    remaining = set(discover_table_names(db, uri))
    missing_protected = PROTECTED_TABLES.intersection(table_names) - remaining
    if missing_protected:
        sys.exit(f"Verification failed: protected tables missing: {sorted(missing_protected)}")

    expected_retained = {name for _, name in retained_backups}
    missing_retained = expected_retained - remaining
    if missing_retained:
        sys.exit(f"Verification failed: retained backups missing: {sorted(missing_retained)}")

    still_present = to_drop.intersection(remaining)
    if still_present:
        sys.exit(f"Verification failed: tables were not removed: {sorted(still_present)}")

    print(f"\nDeleted {len(to_drop)} tables.")
    print("Verification passed: protected tables and retained backups remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
