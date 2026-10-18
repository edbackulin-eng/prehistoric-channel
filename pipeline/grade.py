"""The channel's grade: a colourful night, not a black screen.

The first hook was graded the way the style preset asked - low-key, deep blacks,
muted colour - and on a phone it read as a dark rectangle. Night is still the
look, but lit night: neon teal and magenta, amber street light, the colour a
monitor throws on a wall, and shadows that keep their detail instead of going
to pure black.

Every still and every stock frame passes through grade() before it is moved or
annotated, so the whole film sits in one exposure range. The finished video is
then measured: brightness() samples a frame every two seconds, and the editor's
autofix lifts the picture if it still comes out under the floor.
"""
import subprocess

import numpy as np
from PIL import Image

# Mean luma, 0-255, measured on the frame as the viewer sees it.
TARGET_MEAN = 100          # what grade() aims a frame at
FRAME_FLOOR = 45           # below this a single frame counts as too dark
VIDEO_MEAN_MIN = 70        # the whole film must average at least this
DARK_SHARE_MAX = 0.10      # and at most this share of frames may sit under the floor
BLACK_LIFT = 10            # the darkest a shadow may go: no pure black
SATURATION = 1.25


def mean_luma(im):
    a = np.asarray(im.convert("L").resize((320, 180)), np.float32)
    return float(a.mean())


def grade(im):
    """Lift exposure toward TARGET_MEAN, keep contrast, warm the highlights and
    push teal into the shadows; never crush to black."""
    im = im.convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    luma = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    # Stretch first: the frame's own darkest and brightest points to the full
    # range, so lifting it afterwards adds light rather than grey haze.
    lo, hi = np.percentile(luma, 0.5), np.percentile(luma, 99.7)
    if hi - lo > 0.08:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0, 1)
    luma = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    m = float(luma.mean())
    # A gamma that brings the mean toward the target, bounded so a bright frame
    # is only nudged and a very dark one is lifted without washing out.
    want = TARGET_MEAN / 255.0
    g = np.log(want) / np.log(max(1e-3, min(0.95, m)))
    g = float(np.clip(g, 0.58, 1.10))
    a = a ** g
    # S-curve around the middle so the lift keeps its contrast.
    a = a + 0.22 * (a - 0.5) * (1 - np.abs(2 * a - 1))
    # Split tone: teal into the shadows, amber into the highlights.
    l = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    shadow = np.clip(1 - l * 2.2, 0, 1)
    high = np.clip((l - 0.6) * 2.5, 0, 1)
    a = a + shadow * np.array([-0.015, 0.025, 0.035]) + high * np.array([0.035, 0.015, -0.02])
    # Saturation, around each pixel's own luma.
    l = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    a = l + (a - l) * SATURATION
    lift = BLACK_LIFT / 255.0
    a = lift + np.clip(a, 0, 1) * (1 - lift)
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def brightness(video, every_s=2.0):
    """Mean luma of a frame every `every_s` seconds: {mean, p10, dark_share, ok}."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                        "-vf", f"fps=1/{every_s},scale=160:90,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    n = len(a) // (160 * 90)
    if not n:
        return {"frames": 0, "ok": False, "mean": 0.0, "p10": 0.0, "dark_share": 1.0}
    means = a[: n * 160 * 90].reshape(n, -1).mean(axis=1)
    out = {"frames": int(n), "mean": round(float(means.mean()), 1),
           "p10": round(float(np.percentile(means, 10)), 1),
           "dark_share": round(float((means < FRAME_FLOOR).mean()), 3),
           "floor": FRAME_FLOOR, "mean_min": VIDEO_MEAN_MIN}
    out["ok"] = out["mean"] >= VIDEO_MEAN_MIN and out["dark_share"] <= DARK_SHARE_MAX
    return out
