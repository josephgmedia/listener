"""
Listener — Calendar Watcher
Monitors Google Calendar via API and sends a Windows toast before meetings start.

Setup:
  1. Place credentials.json (from Google Cloud Console) next to this script
  2. pip install google-auth-oauthlib google-api-python-client
  3. Run once manually to authorise — browser will open, sign in, done
  4. Add calendar_watcher_start.bat to your Windows Startup folder
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

CHECK_INTERVAL = 30    # seconds between calendar checks
NOTIFY_BEFORE  = 90    # notify when meeting is within this many seconds

CREDS_FILE    = Path(__file__).parent / "credentials.json"
TOKEN_FILE    = Path(__file__).parent / "calendar_token.json"
NOTIFIED_FILE = Path(__file__).parent / "calendar_notified.json"
LAUNCH_BAT    = Path(__file__).parent / "listener_open.bat"

# ── Windows toast ─────────────────────────────────────────────────────────────
def open_listener():
    subprocess.Popen(
        [str(LAUNCH_BAT)],
        creationflags=0x08000000  # CREATE_NO_WINDOW
    )
    import time, webbrowser
    time.sleep(3)
    webbrowser.open("http://localhost:8765/desktop")


def send_toast(title, message):
    try:
        from winotify import Notification, audio
        toast = Notification(
            app_id="Listener",
            title=title,
            msg=message,
            duration="long",
            launch=str(LAUNCH_BAT)
        )
        toast.set_audio(audio.Default, loop=False)
        toast.add_actions(label="Open Listener", launch=str(LAUNCH_BAT))
        toast.show()
    except Exception as e:
        print(f"  [toast error] {e}")
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command",
             f"[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); "
             f"[System.Windows.Forms.MessageBox]::Show('{message}', '{title}')"],
        )


# ── Google Calendar auth ──────────────────────────────────────────────────────
def get_calendar_service():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    creds  = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ── Fetch upcoming events ─────────────────────────────────────────────────────
def get_events(service):
    try:
        now      = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(hours=12)).isoformat()

        result = service.events().list(
            calendarId  = "primary",
            timeMin     = time_min,
            timeMax     = time_max,
            singleEvents= True,
            orderBy     = "startTime",
            maxResults  = 20,
        ).execute()

        events = []
        for item in result.get("items", []):
            start = item["start"].get("dateTime")
            if not start:
                continue   # skip all-day events
            start_dt = datetime.fromisoformat(start)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            events.append({
                "summary": item.get("summary", "Meeting"),
                "start":   start_dt,
                "uid":     item.get("id", ""),
            })
        return events
    except Exception as e:
        print(f"  [calendar] fetch error: {e}")
        return []


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    if not CREDS_FILE.exists():
        print("=" * 55)
        print("  ERROR: credentials.json not found.")
        print(f"  Expected: {CREDS_FILE}")
        print("=" * 55)
        input("Press Enter to exit...")
        return

    print("  Authorising with Google Calendar...")
    service = get_calendar_service()
    print(f"  Listener Calendar Watcher started — checking every {CHECK_INTERVAL}s")

    # Load previously notified UIDs so restarts don't re-fire
    import json
    notified   = set(json.loads(NOTIFIED_FILE.read_text()) if NOTIFIED_FILE.exists() else [])
    started_at = datetime.now(timezone.utc)

    while True:
        now    = datetime.now(timezone.utc)
        events = get_events(service)

        for ev in events:
            seconds_until = (ev["start"] - now).total_seconds()
            uid           = ev["uid"] or (ev["summary"] + str(ev["start"]))

            just_started = (now - started_at).total_seconds() < 10
            if 0 < seconds_until <= NOTIFY_BEFORE and uid not in notified and not just_started:
                notified.add(uid)
                mins = int(seconds_until // 60)
                secs = int(seconds_until % 60)
                when = f"{mins}m {secs}s" if mins else f"{secs}s"
                print(f"  Notifying: {ev['summary']} in {when}")
                send_toast(
                    f"Meeting in {when} — open Listener",
                    ev["summary"]
                )

        # Prune old UIDs and persist to disk
        active_uids = {ev["uid"] for ev in events}
        notified    = notified & active_uids
        NOTIFIED_FILE.write_text(json.dumps(list(notified)))

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
