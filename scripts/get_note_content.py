#!/usr/bin/env python3
"""scripts/get_note_content.py
Fetch a single note's content by title and optional creation_date using AppleScript.
Prints JSON: {title, creation_date, modification_date, content}

Usage:
  python scripts/get_note_content.py --title "Shopping" [--creation "2025-11-07T12:34:56Z"]
"""
import subprocess
import argparse
import json


def build_applescript(title: str, creation: str) -> str:
    safe_title = title.replace('"', '"')
    safe_creation = creation or ""
    # Return a tab-separated line: title \t creation_iso \t modification_iso \t body
    return f'''
        on replaceText(theText, searchString, replacementString)
            set AppleScript's text item delimiters to searchString
            set theItems to every text item of theText
            set AppleScript's text item delimiters to replacementString
            set theText to theItems as string
            set AppleScript's text item delimiters to ""
            return theText
        end replaceText

        tell application "Notes"
            set noteList to every note
            repeat with n in noteList
                try
                    if name of n is "{title}" then
                        set nCreation to ((creation date of n) as «class isot» as string)
                        if "{safe_creation}" is not "" and nCreation is not "{safe_creation}" then
                            -- skip
                        else
                            set nMod to ((modification date of n) as «class isot» as string)
                            set nBody to body of n
                            -- sanitize
                            set t1 to my replaceText((name of n), tab, " ")
                            set t2 to my replaceText(t1, return, " ")
                            set t3 to my replaceText(t2, linefeed, " ")
                            set b1 to my replaceText(nBody, tab, " ")
                            set b2 to my replaceText(b1, return, " ")
                            set b3 to my replaceText(b2, linefeed, " ")
                            return t3 & tab & nCreation & tab & nMod & tab & b3
                        end if
                    end if
                end try
            end repeat
        end tell
        return ""
    '''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', required=True, help='Note title')
    parser.add_argument('--creation', default='', help='Creation date (ISO) to disambiguate identical titles')
    args = parser.parse_args()

    script = build_applescript(args.title, args.creation)

    try:
        r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        out = r.stdout.strip()
        if not out:
            print(json.dumps({"error": "Note not found or no output"}))
            return

        try:
            data = json.loads(out)
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError:
            # Fallback: print raw output
            print(json.dumps({"raw": out}))

    except subprocess.CalledProcessError as e:
        print(json.dumps({"error": "osascript failed", "stdout": e.stdout, "stderr": e.stderr}))


if __name__ == '__main__':
    main()


def fetch_note_content(title: str, creation: str = '') -> dict:
    """Programmatic access: fetch a single note by title and optional creation date.

    Returns a dict: {title, creation_date, modification_date, content} or an
    error dict on failure. This mirrors the behavior of the CLI entrypoint but
    returns structured data.
    """
    script = build_applescript(title, creation)
    try:
        r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        out = r.stdout.strip()
        if not out:
            return {"error": "Note not found or no output", "title": title}
        parts = out.split('\t')
        return {
            'title': parts[0] if len(parts) > 0 else title,
            'creation_date': parts[1] if len(parts) > 1 else creation,
            'modification_date': parts[2] if len(parts) > 2 else '',
            'content': parts[3] if len(parts) > 3 else ''
        }
    except subprocess.CalledProcessError as e:
        return {"error": "osascript_failed", "stderr": e.stderr, "stdout": e.stdout, 'title': title}
