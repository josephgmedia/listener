"""
Listener — Calendar Watcher
Monitors Google Calendar and sends a Windows toast 1 minute before meetings start.

Setup:
  1. Paste your Google Calendar iCal URL into ICAL_URL below
  2. pip install icalendar requests
  3. Add calendar_watcher_start.bat to your Windows Startup folder
"""

import time
import subprocess
import requests
from datetime import datetime, timezone
from icalendar import Calendar

# ── Config ────────────────────────────────────────────────────────────────────
# iCal URL is stored in a separate file (calendar.secret) so it never hits GitHub.
# Create that file next to this script and paste your iCal URL as the only line.
import pathlib
_secret = pathlib.Path(__file__).parent / "calendar.secret"
ICAL_URL = _secret.read_text().strip() if _secret.exists() else "PASTE_YOUR_ICAL_URL_HERE"
CHECK_INTERVAL  = 30     # seconds between calendar checks
NOTIFY_BEFORE   = 90     # notify when meeting is within this many seconds
                         # (90s window covers a 30s polling gap so nothing slips through)
LISTENER_URL    = "http://localhost:8765/desktop"

# ── Windows toast (no extra packages — uses PowerShell) ───────────────────────
def send_toast(title, message):
    ps = f"""
$ErrorActionPreference = 'SilentlyContinue'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$nodes = $xml.GetElementsByTagName('text')
$nodes.Item(0).AppendChild($xml.CreateTextNode('{title}')) | Out-Null
$nodes.Item(1).AppendChild($xml.CreateTextNode('{message}')) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Listener').Show($toast)
"""
    subprocess.run(
        ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", ps],
        capture_output=True
    )

# ── Calendar fetch + parse ────────────────────────────────────────────────────
def get_events():
    try:
        resp = requests.get(ICAL_URL, timeout=10)
        resp.raise_for_status()
        cal = Calendar.from_ical(resp.content)
        events = []
        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART")
            if not dtstart:
                continue
            start = dtstart.dt
            # Handle all-day events (date only, no time) — skip them
            if not isinstance(start, datetime):
                continue
            # Ensure timezone-aware
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            events.append({
                "summary": str(component.get("SUMMARY", "Meeting")),
                "start":   start,
                "uid":     str(component.get("UID", "")),
            })
        return events
    except Exception as e:
        print(f"  [calendar] fetch error: {e}")
        return []

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    if ICAL_URL == "PASTE_YOUR_ICAL_URL_HERE":
        print("=" * 55)
        print("  ERROR: No iCal URL set.")
        print("  Open calendar_watcher.py and paste your")
        print("  Google Calendar iCal URL into ICAL_URL.")
        print("=" * 55)
        input("Press Enter to exit...")
        return

    print(f"  Listener Calendar Watcher started — checking every {CHECK_INTERVAL}s")

    notified = set()   # UIDs we've already fired a notification for

    while True:
        now    = datetime.now(timezone.utc)
        events = get_events()

        for ev in events:
            seconds_until = (ev["start"] - now).total_seconds()
            uid           = ev["uid"] or (ev["summary"] + str(ev["start"]))

            if 0 < seconds_until <= NOTIFY_BEFORE and uid not in notified:
                notified.add(uid)
                mins = int(seconds_until // 60)
                secs = int(seconds_until % 60)
                when = f"{mins}m {secs}s" if mins else f"{secs}s"
                print(f"  Notifying: {ev['summary']} in {when}")
                send_toast(
                    f"Meeting in {when} — open Listener",
                    ev["summary"]
                )

        # Clean up old UIDs from notified set to avoid unbounded growth
        cutoff = now.timestamp()
        notified = {uid for uid in notified
                    if any(uid == (ev["uid"] or ev["summary"] + str(ev["start"]))
                           and ev["start"].timestamp() > cutoff - 3600
                           for ev in events)}

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
