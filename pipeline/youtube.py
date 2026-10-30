"""YouTube for the German prehistoric channel: upload, schedule, thumbnail, subtitles.

Every upload goes up private; until config "release.manual_left" is spent it
stays unscheduled until approved (approve.yml), after that it gets a publishAt
on the nearest free slot of the channel's week - Wednesday and Saturday, 17:00 Europe/Berlin (config/channel.json
"schedule"; summer and winter time follow the zone) - at least MIN_LEAD away.
Every video declares containsSyntheticMedia=true and madeForKids=false, German
as its language, and carries German subtitles.

A video is never uploaded twice: before an upload the channel itself is asked
whether it has the episode (its title or its urzeit-<slug> tag); the moment
YouTube returns an id it is written to episodes/<slug>/uploaded.json and
committed (automation/gitsave.py); an episode with that file is never uploaded.
On 30.10 one episode went up twice: the subtitles failed after the upload, the
step failed before the record was committed, and a rerun uploaded it again.

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
# captions.insert needs youtube.force-ssl: without it the upload works but the subtitles fail (30.10)
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube",
          "https://www.googleapis.com/auth/youtube.force-ssl"]
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
    """publish_at None: private and unscheduled, until the episode is approved."""
    st = {"privacyStatus": CFG["privacy"], "selfDeclaredMadeForKids": False,
          "containsSyntheticMedia": bool(CFG["contains_synthetic_media"])}
    if publish_at is not None:
        st["publishAt"] = publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"snippet": {"title": title[:100], "description": description[:5000], "tags": own_tags(tags)[:30],
                        "categoryId": CATEGORY_EDUCATION, "defaultLanguage": "de", "defaultAudioLanguage": "de"},
            "status": st}


def channel_videos(yt):
    """Every video of the channel: id, title, tags, privacy, publishAt (the uploads playlist)."""
    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    up = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, tok = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=up, maxResults=50, pageToken=tok).execute()
        ids += [i["contentDetails"]["videoId"] for i in r["items"]]
        tok = r.get("nextPageToken")
        if not tok:
            break
    out = []
    for i in range(0, len(ids), 50):
        for v in yt.videos().list(part="snippet,status", id=",".join(ids[i:i + 50])).execute()["items"]:
            out.append({"id": v["id"], "title": v["snippet"]["title"], "tags": v["snippet"].get("tags") or [],
                        "privacy": v["status"]["privacyStatus"], "publish_at": v["status"].get("publishAt")})
    return out


def slug_tag(slug):
    return f"urzeit-{slug}"


def already_on_channel(slug, title, yt, ignore=()):
    """The episode's video if the channel has one (same title or the episode's tag), else None.
    `ignore`: ids of rejected videos that a rebuild replaces."""
    for v in channel_videos(yt):
        if v["id"] in ignore:
            continue
        if v["title"].strip().casefold() == title[:100].strip().casefold() or slug_tag(slug) in v["tags"]:
            return v
    return None


def set_schedule(vid, publish_at, yt=None):
    """Private with publishAt; publish_at None: private and unscheduled. Returns the new status."""
    yt = yt or service()
    st = yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]
    new = {k: st[k] for k in ("license", "embeddable", "publicStatsViewable", "selfDeclaredMadeForKids",
                              "containsSyntheticMedia") if k in st}
    new.update({"privacyStatus": "private",
                "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ") if publish_at else None})
    yt.videos().update(part="status", body={"id": vid, "status": new}).execute()
    return yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]


def upload(slug, release_dir, title, description, tags, publish_at, yt=None, ignore=(), on_id=None):
    """Upload release_dir/video.mp4 with thumbnail and German subtitles. Returns the id.

    One episode, one video: refused when episodes/<slug>/uploaded.json exists or the
    channel already has the episode (title or tag); the id is written and handed to
    on_id (which commits it) the moment YouTube returns it, before anything else can
    fail. Thumbnail and subtitles are extras: a failure there is logged and kept in
    uploaded.json, never raised, so no rerun of the step can upload the video again."""
    from googleapiclient.http import MediaFileUpload
    marker = ROOT / "episodes" / slug / "uploaded.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        log.warning("%s already uploaded as %s; not uploading again", slug, saved["youtube_id"])
        return saved["youtube_id"]
    yt = yt or service()
    there = already_on_channel(slug, title, yt, ignore)
    if there:
        log.warning("%s is on the channel already as %s (%s); not uploading again", slug, there["id"],
                    there["title"])
        marker.write_text(json.dumps({"youtube_id": there["id"], "publish_at": there["publish_at"],
                                      "found_on_channel": True}), encoding="utf-8")
        if on_id:
            on_id(there["id"])
        return there["id"]
    release_dir = Path(release_dir)
    media = MediaFileUpload(str(release_dir / "video.mp4"), chunksize=CHUNK_MB * 1024 * 1024, resumable=True,
                            mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body(title, description, list(tags) + [slug_tag(slug)],
                                                              publish_at), media_body=media)
    response, last = None, -1
    while response is None:
        status, response = req.next_chunk()
        if status and int(status.progress() * 100) >= last + 10:
            last = int(status.progress() * 100)
            log.info("  upload %d%%", last)
    vid = response["id"]
    marker.write_text(json.dumps({"youtube_id": vid, "publish_at": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                                  if publish_at else None}), encoding="utf-8")
    log.info("uploaded https://youtu.be/%s (%s)", vid, publish_at.isoformat() if publish_at else "private, unscheduled")
    if on_id:
        on_id(vid)
    extras = {}
    thumb = release_dir / "thumbnail.jpg"
    if thumb.exists():
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
            extras["thumbnail"] = "set"
        except Exception as e:                                       # noqa: BLE001
            extras["thumbnail"] = f"failed: {str(e)[:200]}"
    srt = release_dir / "subtitles_de.srt"
    if srt.exists():
        try:
            yt.captions().insert(part="snippet", body={"snippet": {"videoId": vid, "language": "de",
                                                                   "name": "Deutsch", "isDraft": False}},
                                 media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream")).execute()
            extras["subtitles"] = "set"
        except Exception as e:                                       # noqa: BLE001
            extras["subtitles"] = f"failed: {str(e)[:200]}"
    bad = {k: v for k, v in extras.items() if v != "set"}
    if bad:
        log.warning("%s uploaded as %s, but: %s", slug, vid, bad)
    saved = json.loads(marker.read_text(encoding="utf-8"))
    marker.write_text(json.dumps({**saved, "extras": extras}), encoding="utf-8")
    return vid
