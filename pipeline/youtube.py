"""YouTube for the German prehistoric channel: upload, schedule, thumbnail, subtitles.

Every upload goes up private with a publishAt on the nearest free slot of the
channel's week - Wednesday and Saturday, 17:00 Europe/Berlin (config/channel.json
"schedule"; summer and winter time follow the zone) - at least MIN_LEAD away.
Every video declares containsSyntheticMedia=true and madeForKids=false, German
as its language, and carries German subtitles.

A video is never uploaded twice: the moment YouTube returns an id it is written
to episodes/<slug>/uploaded.json, and the upload refuses to run for an episode
that already has one (the pattern of closed-tabs, after the stoicism channel
uploaded one video twice).

Scopes: youtube.upload to upload, youtube (full) for the thumbnail, captions and
schedule. The token comes from the YT_TOKEN_JSON secret.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from core.logger import get_logger

log = get_logger("youtube")

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["schedule"]
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
CATEGORY_EDUCATION = "27"
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
SLOT_WEEKDAYS = tuple(DAYS[d] for d in CFG["weekdays"])
SLOT_TIME = tuple(int(x) for x in CFG["time"].split(":"))
TZ = ZoneInfo(CFG["tz"])
MIN_LEAD = timedelta(minutes=30)
CHUNK_MB = 8


def available():
    return bool(os.environ.get("YT_TOKEN_JSON", "").strip())


def _credentials():
    from google.oauth2.credentials import Credentials
    data = json.loads(os.environ["YT_TOKEN_JSON"])
    return Credentials(token=None, refresh_token=data["refresh_token"], client_id=data["client_id"],
                       client_secret=data["client_secret"],
                       token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
                       scopes=data.get("scopes") or SCOPES)


def service():
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=_credentials(), cache_discovery=False)


def authorize(client_secret, token_out):
    """Local browser consent; writes the refresh token. Run once, by hand."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    data = {"refresh_token": creds.refresh_token, "client_id": creds.client_id,
            "client_secret": creds.client_secret, "token_uri": creds.token_uri, "scopes": SCOPES}
    Path(token_out).write_text(json.dumps(data), encoding="utf-8")
    return data


def next_slot(taken=(), now=None):
    """The nearest free slot (UTC), Wednesday or Saturday 17:00 in Berlin."""
    now = now or datetime.now(timezone.utc)
    taken_days = set()
    for t in taken:
        t = datetime.fromisoformat(t.replace("Z", "+00:00")) if isinstance(t, str) else t
        taken_days.add(t.astimezone(TZ).date())
    day = now.astimezone(TZ).replace(hour=SLOT_TIME[0], minute=SLOT_TIME[1], second=0, microsecond=0)
    for _ in range(120):
        if day.weekday() in SLOT_WEEKDAYS and day - now >= MIN_LEAD and day.date() not in taken_days:
            return day.astimezone(timezone.utc)
        day = (day + timedelta(days=1)).replace(hour=SLOT_TIME[0], minute=SLOT_TIME[1])
    raise RuntimeError("no free slot in the next 120 days")


def slots(n, taken=(), now=None):
    out, t = [], list(taken)
    for _ in range(n):
        s = next_slot(t, now)
        out.append(s)
        t.append(s)
    return out


def own_tags(tags):
    """The tags without any other channel's, broadcaster's or film's name (config schedule.tag_brands)."""
    import re
    brands = [re.sub(r"[^a-z0-9äöüß]", "", b.lower()) for b in CFG.get("tag_brands", [])]
    out = []
    for t in tags:
        words = re.findall(r"[a-z0-9äöüß.]+", t.lower())
        flat = re.sub(r"[^a-z0-9äöüß]", "", t.lower())
        if any(b and (b == flat or (len(b) > 4 and b in flat) or b in [re.sub(r"\.", "", w) for w in words])
               for b in brands):
            log.info("tag dropped (a brand): %s", t)
            continue
        out.append(t)
    return out


def body(title, description, tags, publish_at):
    return {"snippet": {"title": title[:100], "description": description[:5000], "tags": own_tags(tags)[:30],
                        "categoryId": CATEGORY_EDUCATION, "defaultLanguage": "de", "defaultAudioLanguage": "de"},
            "status": {"privacyStatus": CFG["privacy"], "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "selfDeclaredMadeForKids": False,
                       "containsSyntheticMedia": bool(CFG["contains_synthetic_media"])}}


def upload(slug, release_dir, title, description, tags, publish_at, yt=None):
    """Upload release_dir/video.mp4 with thumbnail and German subtitles. Returns the id."""
    from googleapiclient.http import MediaFileUpload
    marker = ROOT / "episodes" / slug / "uploaded.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        log.warning("%s already uploaded as %s; not uploading again", slug, saved["youtube_id"])
        return saved["youtube_id"]
    yt = yt or service()
    release_dir = Path(release_dir)
    media = MediaFileUpload(str(release_dir / "video.mp4"), chunksize=CHUNK_MB * 1024 * 1024, resumable=True,
                            mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body(title, description, tags, publish_at),
                             media_body=media)
    response, last = None, -1
    while response is None:
        status, response = req.next_chunk()
        if status and int(status.progress() * 100) >= last + 10:
            last = int(status.progress() * 100)
            log.info("  upload %d%%", last)
    vid = response["id"]
    marker.write_text(json.dumps({"youtube_id": vid, "publish_at": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")}),
                      encoding="utf-8")
    log.info("uploaded https://youtu.be/%s for %s", vid, publish_at.isoformat())
    thumb = release_dir / "thumbnail.jpg"
    if thumb.exists():
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
    srt = release_dir / "subtitles_de.srt"
    if srt.exists():
        yt.captions().insert(part="snippet", body={"snippet": {"videoId": vid, "language": "de",
                                                               "name": "Deutsch", "isDraft": False}},
                             media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream")).execute()
    return vid
