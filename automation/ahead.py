"""Episodes prepared ahead (owner 10.11: two videos a week, a buffer of two built episodes).

    python automation/ahead.py

While episode N is on FLUX, LTX or its render, the steps that need no GPU run for what comes next:
  1. scripts: the next three queued topics without episodes/<slug>/script.txt are written and
     fact-checked (automation/write_episode.py), each when the day's text quota allows;
  2. the next topic (the first queued one): its plan, the CPU prefilter, the species check of its
     stock clips and, once every beat is judged, the stand-ins of living animals - exactly the steps
     of episode.yml, which then finds them done.
The image quota is shared by days: the next topic's check takes only what is left after the episode
being built has its own needs (its FLUX check and its render, automation/llm_budget.py), and a check
cut short goes on the next day. Nothing here defers the episode being built (state/deferred.json is
left alone) and nothing touches the GPU. ahead.yml commits each step as it is done.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# (not automation/ itself on the path: its watchdog.py would hide gpu/kernels/watchdog.py)
from automation import gitsave  # noqa: E402
from core import llm_usage  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("ahead")
# owner 12.11: two videos a week (Sun, Wed) and nothing built beyond them - one script ahead, and the
# next topic's plan and species check only once fewer than `buffer` episodes are ahead or being built
SCRIPTS_AHEAD = 1


def queue():
    return json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))["topics"]


def image_room():
    from automation import llm_budget as b
    return max(0, llm_usage.left(b.FLASH)["requests"]) + max(0, llm_usage.left(b.LITE)["requests"])


def reserve_for(slug):
    """What the episode being built still needs of today's images: its FLUX check, its render; and
    what an episode being rebuilt still has to have judged."""
    from automation import llm_budget as b
    extra = 0
    for s in rebuilding():
        try:
            extra += b.image_need(s)
        except Exception:                                    # noqa: BLE001
            extra += 120
    if not slug:
        return extra
    try:
        return extra + b.Q["per_episode"]["flux_check"]["image_requests"] + b.image_need(slug)
    except Exception as e:                                   # noqa: BLE001
        log.info("reserve for %s not known (%s): 120", slug, e)
        return 120


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT).returncode


def save(slug, what):
    gitsave.save([ROOT / "episodes" / slug, ROOT / "state" / "llm_usage", ROOT / "topics" / "queue.json"], f"Ahead: {what} for {slug}")


def scripts(topics):
    from automation import llm_budget as b
    pe = b.Q["per_episode"]["script"]
    from automation import topics as _topics
    q = _topics.load()
    by = {x["slug"]: x for x in q["topics"]}
    usable = []
    for t in topics:
        x = by.get(t["slug"], t)
        if "material" not in x:                  # owner 12.11: material first, then a script
            x["material"] = _topics.material(x)
            if x["material"].get("ok") is False:
                x["status"] = "skipped-material"
            _topics.save(q)
        if x.get("status") == "queued":
            usable.append(x)
        if len(usable) >= SCRIPTS_AHEAD:
            break
    for t in usable[:SCRIPTS_AHEAD]:
        slug = t["slug"]
        if (ROOT / "episodes" / slug / "script.txt").exists():
            continue
        ok, why = b.text_ok(pe["text_requests"], pe["text_tokens"])
        if not ok:
            log.info("scripts ahead stop at %s: %s", slug, why)
            return
        log.info("writing ahead: %s (%s)", slug, why)
        if run(ROOT / "automation" / "write_episode.py", slug) == 0:
            save(slug, "script")
        else:
            log.info("%s not written now; the next script waits for the next run", slug)
        # 11.11: three scripts in one night spent Groq's day of tokens and the flash-lite requests the
        # species checks need - one script a run (four runs a day), and none after a failure
        return


def prepare(slug, building):
    from automation import llm_budget as b
    from automation import plan_episode
    from pipeline import speciescheck
    ep = ROOT / "episodes" / slug
    if not (ep / "script.txt").exists():
        return "no script yet"
    if plan_episode.stale(slug):
        ok, why = b.text_ok(b.Q["per_episode"]["plan"]["text_requests"], b.Q["per_episode"]["plan"]["text_tokens"])
        if not ok:
            return f"plan waits: {why}"
        if run(ROOT / "automation" / "plan_episode.py", "--if-stale", slug) != 0:
            return "plan failed"
        save(slug, "plan")
    speciescheck.prefilter_pass(slug)
    save(slug, "prefilter")
    need = b.image_need(slug, kinds=("real",), fixed=False)
    room, keep = image_room(), reserve_for(building)
    log.info("%s: species check needs ~%d; %d image requests left today, %d kept for %s", slug, need, room, keep,
             building or "-")
    if need and room - keep < 1:
        return f"species check waits for quota (~{need} needed; {room} left, {keep} kept for {building})"
    if need:
        speciescheck.check(slug, ROOT / "work" / "none", ROOT / "work" / slug / "footage_check", kinds=("real",))
        save(slug, "species check")
        rep = json.loads((ROOT / "work" / slug / "footage_check" / "species_check.json").read_text(encoding="utf-8"))
        left = sum(1 for r in rep["rows"] if r.get("action") == "unknown"
                   and any(c.get("verdict") == "unknown" for c in r["candidates"]))
        if left:
            return f"species check: {left} beats still unjudged, the rest after the next reset"
    if run(ROOT / "automation" / "plan_episode.py", "--standby", slug) == 0:
        save(slug, "stand-ins")
    return "prepared: plan, species check and stand-ins done; FLUX when it is its turn"


def rebuilding():
    """Episodes whose QA failed and that are being rebuilt (release.ahead counts them): their species
    check comes before the next topic's."""
    from automation import release
    return [x["slug"] for x in release.ahead() if x["status"] == "qa-failed"]


def main():
    from automation import release
    topics = queue()
    building = next((t["slug"] for t in topics if t.get("status") == "building"), None)
    nxt = [t for t in topics if t.get("status") == "queued"]
    scripts(nxt)
    full = len(release.ahead()) + (1 if building else 0)
    if full >= json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["buffer"]:
        log.info("nothing prepared ahead: %d episodes ahead or being built (%s, building %s) fill the schedule",
                 full, [x["slug"] for x in release.ahead()], building or "-")
        return 0
    if nxt:
        log.info("next after %s: %s - %s", building or "nothing", nxt[0]["slug"], prepare(nxt[0]["slug"], building))
    return 0


if __name__ == "__main__":
    sys.exit(main())
