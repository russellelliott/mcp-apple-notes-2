#!/usr/bin/env python3
"""scripts/print_note_titles.py
Fetch note titles using AppleScript (avoids heavy imports) and print them.

Usage: python scripts/print_note_titles.py [--limit N]
"""
import subprocess
import argparse
import time

# Try to import py-applescript; fall back to subprocess if not available
try:
    from applescript import AppleScript
    HAS_PY_APPLESCRIPT = True
except ImportError:
    HAS_PY_APPLESCRIPT = False


def _run_applescript(script: str) -> str:
    """Execute AppleScript using py-applescript if available, else subprocess."""
    if HAS_PY_APPLESCRIPT:
        try:
            as_script = AppleScript(script)
            result = as_script.run()
            return result if result else ""
        except Exception as e:
            print(f"[!] py-applescript failed: {e}, falling back to osascript", flush=True)
    
    # Fallback to subprocess
    try:
        r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        return r.stdout
    except subprocess.CalledProcessError as e:
        print(f"[!] osascript failed: {e.stderr}", flush=True)
        return ""


def build_applescript(limit: int) -> str:
    apple_limit = limit if limit is not None else 0
    return f"""
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
                    set c to (creation date of n) as string
                    set m to (modification date of n) as string

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Limit number of notes to fetch (0 = all)')
    args = parser.parse_args()

    script = build_applescript(args.limit)

    start = time.time()
    out = _run_applescript(script).strip()
    
    if not out:
        print('No output from osascript')
        return

    lines = out.splitlines()
    if not lines:
        print('No notes returned (empty output)')
        return

    for i, line in enumerate(lines, 1):
        parts = line.split('\t')
        title = parts[0] if len(parts) > 0 else '<untitled>'
        creation = parts[1] if len(parts) > 1 else ''
        modification = parts[2] if len(parts) > 2 else ''
        print(f"{i}. {title} (created: {creation}, modified: {modification})")

    elapsed = time.time() - start
    print(f"\nFetched {len(lines)} titles in {elapsed:.2f}s")


if __name__ == '__main__':
    main()


def fetch_titles(limit: int) -> list:
    """Programmatic access: return a list of dicts with title and creation_date.

    This reuses the same AppleScript used by the CLI entrypoint.
    """
    script = build_applescript(limit)
    out = _run_applescript(script).strip()
    if not out:
        return []
    items = []
    for line in out.splitlines():
        parts = line.split('\t')
        title = parts[0] if len(parts) > 0 else '<untitled>'
        creation = parts[1] if len(parts) > 1 else ''
        modification = parts[2] if len(parts) > 2 else ''
        items.append({'title': title, 'creation_date': creation, 'modification_date': modification})
    return items
