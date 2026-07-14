"""
Listener — Add a Google Calendar Account

Run this once for each Google account you want Listener to read. Your existing
account at calendar_token.json keeps working untouched — this just adds another.

  python add_calendar_account.py            # interactive — asks for a label
  python add_calendar_account.py work       # named account
  python add_calendar_account.py personal   # another named account

The first time you run it, a browser window will open for Google sign-in.
Listener will then automatically check every connected account for meetings.

Setup requirements (one time):
  1. credentials.json must already exist in this folder (same file you used
     to set up calendar_watcher.py)
  2. pip install google-auth google-auth-oauthlib google-api-python-client
"""

import sys
import re
from pathlib import Path

CREDS_FILE  = Path(__file__).parent / "credentials.json"
SCOPES      = ["https://www.googleapis.com/auth/calendar.readonly"]


def main():
    if not CREDS_FILE.exists():
        print("=" * 60)
        print("  ERROR: credentials.json not found.")
        print(f"  Expected: {CREDS_FILE}")
        print()
        print("  Get one from Google Cloud Console (OAuth 2.0 client ID).")
        print("=" * 60)
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print("=" * 60)
        print("  Missing dependencies. Install with:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        print("=" * 60)
        sys.exit(1)

    # Label for the new account (used in the filename — kept short and safe)
    if len(sys.argv) > 1:
        label = sys.argv[1].strip()
    else:
        label = input("Label for this account (e.g. 'work', 'personal'): ").strip()
    if not label:
        print("Label can't be empty. Aborting.")
        sys.exit(1)

    label_safe = re.sub(r"[^a-z0-9_-]", "_", label.lower())
    token_file = Path(__file__).parent / f"calendar_token_{label_safe}.json"

    if token_file.exists():
        ans = input(f"  {token_file.name} already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print()
    print("  Opening Google sign-in in your browser…")
    print("  Sign in with the account you want to add.")
    print()

    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    token_file.write_text(creds.to_json())

    # Confirm which email we just authorised — useful when you have several
    try:
        service = build("calendar", "v3", credentials=creds)
        cal     = service.calendarList().get(calendarId="primary").execute()
        email   = cal.get("id", "unknown")
    except Exception:
        email = "unknown"

    print()
    print("=" * 60)
    print(f"  ✓ Connected: {email}")
    print(f"  ✓ Saved to:  {token_file.name}")
    print("=" * 60)
    print()
    print("  Restart Listener (close cmd, double-click launch.bat).")
    print("  This account will now be checked alongside any others.")


if __name__ == "__main__":
    main()
