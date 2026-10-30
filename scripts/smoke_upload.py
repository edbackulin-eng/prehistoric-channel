"""Smoke test of the one-episode-one-video rule, against a fake YouTube (no network).

  1. an upload whose subtitles fail (as on 30.10, 403) still returns the id, and
     the id is handed over for committing before the subtitles are tried;
  2. a second upload of the same episode - the rerun that made the duplicate -
     uploads nothing;
  3. without uploaded.json (lost record) the channel is asked, finds the video by
     title or tag, and nothing is uploaded;
  4. a rejected video is ignored, so the rebuild may upload exactly once;
  5. unscheduled uploads (release.manual_left) carry no publishAt.
Returns a list of (name, ok, detail).
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Exec:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeYT:
    def __init__(self, videos=None, fail_captions=True):
        self.videos_ = list(videos or [])
        self.inserts = []
        self.fail_captions = fail_captions
        self.order = []

    def channels(self):
        return type("C", (), {"list": lambda s, **k: _Exec(lambda: {"items": [
            {"contentDetails": {"relatedPlaylists": {"uploads": "UU"}}}]})})()

    def playlistItems(self):
        vs = self.videos_
        return type("P", (), {"list": lambda s, **k: _Exec(lambda: {"items": [
            {"contentDetails": {"videoId": v["id"]}} for v in vs]})})()

    def videos(self):
        yt = self

        class V:
            def list(self, id, **k):
                ids = id.split(",")
                return _Exec(lambda: {"items": [{"id": v["id"], "snippet": {"title": v["title"], "tags": v.get("tags", [])},
                                                 "status": {"privacyStatus": "private", "publishAt": v.get("publishAt")}}
                                                for v in yt.videos_ if v["id"] in ids]})

            def insert(self, part, body, media_body):
                class Req:
                    def next_chunk(s):
                        vid = f"fake{len(yt.inserts)}"
                        yt.inserts.append(body)
                        yt.order.append("insert")
                        yt.videos_.append({"id": vid, "title": body["snippet"]["title"], "tags": body["snippet"]["tags"],
                                           "publishAt": body["status"].get("publishAt")})
                        return None, {"id": vid}
                return Req()
        return V()

    def thumbnails(self):
        yt = self
        return type("T", (), {"set": lambda s, **k: _Exec(lambda: yt.order.append("thumb"))})()

    def captions(self):
        yt = self

        def boom():
            yt.order.append("captions")
            if yt.fail_captions:
                raise RuntimeError("HttpError 403 insufficient authentication scopes")
        return type("Cap", (), {"insert": lambda s, **k: _Exec(boom)})()


def run(work):
    from pipeline import youtube
    out = []
    slug = "smoke-dup"
    ep = ROOT / "episodes" / slug
    rel = Path(work) / "dup_release"
    shutil.rmtree(ep, ignore_errors=True)
    rel.mkdir(parents=True, exist_ok=True)
    ep.mkdir(parents=True)
    try:
        (rel / "video.mp4").write_bytes(b"\0" * 1024)
        (rel / "subtitles_de.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHallo\n", encoding="utf-8")
        yt, got = FakeYT(), []

        def on_id(vid):
            got.append((vid, list(yt.order)))
        vid = youtube.upload(slug, rel, "Smoke Titel", "d", ["Urzeit"], None, yt=yt, on_id=on_id)
        marker = json.loads((ep / "uploaded.json").read_text(encoding="utf-8"))
        out.append(("upload: id kept before the subtitles fail", vid == "fake0" and got == [("fake0", ["insert"])]
                    and "failed" in marker["extras"]["subtitles"], str((got, marker.get("extras")))))
        out.append(("upload: no date while approval is pending", "publishAt" not in yt.inserts[0]["status"]
                    and marker["publish_at"] is None, str(yt.inserts[0]["status"])))
        youtube.upload(slug, rel, "Smoke Titel", "d", [], None, yt=yt)
        out.append(("rerun uploads nothing", len(yt.inserts) == 1, str(len(yt.inserts))))
        (ep / "uploaded.json").unlink()
        again = youtube.upload(slug, rel, "smoke titel ", "d", [], None, yt=yt)
        out.append(("lost record: the channel is asked, nothing uploaded", len(yt.inserts) == 1 and again == "fake0",
                    str((len(yt.inserts), again))))
        (ep / "uploaded.json").unlink()
        renamed = FakeYT([{"id": "old1", "title": "Anderer Titel", "tags": [youtube.slug_tag(slug)]}])
        youtube.upload(slug, rel, "Neuer Titel", "d", [], None, yt=renamed)
        out.append(("found by the episode's tag", not renamed.inserts, str(len(renamed.inserts))))
        (ep / "uploaded.json").unlink()
        rebuilt = FakeYT([{"id": "old1", "title": "Smoke Titel", "tags": [youtube.slug_tag(slug)]}])
        v2 = youtube.upload(slug, rel, "Smoke Titel", "d", [], None, yt=rebuilt, ignore=["old1"])
        v3 = youtube.upload(slug, rel, "Smoke Titel", "d", [], None, yt=rebuilt, ignore=["old1"])
        out.append(("rejected video ignored: the rebuild uploads once", len(rebuilt.inserts) == 1 and v2 == v3,
                    str((len(rebuilt.inserts), v2, v3))))
    finally:
        shutil.rmtree(ep, ignore_errors=True)
    from automation import release
    ok, why = release.qa_passed(rel)
    out.append(("no QA report, no upload", not ok and why, str(why)))
    return out
