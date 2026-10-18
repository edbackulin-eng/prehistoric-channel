"""2.5D: a still split into depth layers and filmed with a slow moving camera.

Layers come from the Kaggle job (kaggle/recon_job.py): the image, a soft
foreground alpha cut from Depth Anything V2's depth map, and a background in
which LaMa has painted over the foreground (plus a margin). The camera moves
the background less than the foreground, which is all parallax is.

Stretch check, before a single frame is rendered (check()):
  1. coverage - neither layer may ever leave a gap at the frame edge;
  2. reveal   - the most the foreground slides over the background must stay
                inside the margin LaMa painted, or raw stretched pixels show;
  3. sanity   - a foreground of under 2 % or over 60 % of the frame is a
                failed depth split, and the shot falls back to one layer.
A move that fails 1 or 2 is scaled down until it passes; the report records it.
"""
import math

import cv2
import numpy as np

SRC_W, SRC_H = 4224, 2376
# LaMa filled the hole dilated by 22 px at 1408 wide, i.e. 66 px at 4224;
# a 0.75 safety factor leaves room for the soft alpha edge.
REVEAL_LIMIT_PX = 66 * 0.75

# (zoom_fg, zoom_bg, dx_fg, dx_bg, dy_fg, dy_bg) over the whole shot, in
# fractions of the frame width/height. The background moves a third as much.
MOVES = {
    "push_in":      (0.075, 0.035, 0.0, 0.0, 0.0, 0.0),
    "pull_out":     (-0.07, -0.03, 0.0, 0.0, 0.0, 0.0),
    "truck_right":  (0.03, 0.015, 0.024, 0.008, 0.0, 0.0),
    "truck_left":   (0.03, 0.015, -0.024, -0.008, 0.0, 0.0),
    "crane_up":     (0.03, 0.015, 0.0, 0.0, -0.02, -0.007),
    "push_in_left": (0.06, 0.03, -0.012, -0.004, 0.0, 0.0),
}


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t)))


def _affine(zoom, dx, dy, out_w, out_h):
    """Source->output matrix: a centred crop of the source, zoomed and shifted."""
    # source px -> output px; at zoom 0 the window shows the central
    # 3840x2160 of the 4224x2376 source (scaled for a 1440p render)
    s = (out_w / 3840.0) * (1.0 + zoom)
    cx, cy = SRC_W / 2 + dx * SRC_W, SRC_H / 2 + dy * SRC_H
    return np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], dtype=np.float64)


def _params(move, t, amp):
    zf, zb, xf, xb, yf, yb = MOVES[move]
    e = ease(t)
    if zf < 0:                       # pull out: start close, end wide
        zf0, zb0 = -zf, -zb
        return ((zf0 + zf * e) * amp, (zb0 + zb * e) * amp, xf * (e - .5) * amp, xb * (e - .5) * amp,
                yf * (e - .5) * amp, yb * (e - .5) * amp)
    return zf * e * amp, zb * e * amp, xf * (e - .5) * amp, xb * (e - .5) * amp, \
        yf * (e - .5) * amp, yb * (e - .5) * amp


def _covers(m, out_w, out_h):
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0, 1], [out_w, 0, 1], [0, out_h, 1], [out_w, out_h, 1]], dtype=np.float64)
    pts = corners @ inv.T
    return (pts[:, 0].min() >= 0 and pts[:, 1].min() >= 0 and pts[:, 0].max() <= SRC_W
            and pts[:, 1].max() <= SRC_H)


def check(move, fg_area, frames=24, out_w=3840, out_h=2160):
    """(amplitude, layered, report): the largest safe fraction of the move."""
    layered = 0.02 <= fg_area <= 0.60
    for amp in (1.0, 0.8, 0.6, 0.45, 0.3):
        ok, worst = True, 0.0
        for k in range(frames + 1):
            zf, zb, xf, xb, yf, yb = _params(move, k / frames, amp)
            mf, mb = _affine(zf, xf, yf, out_w, out_h), _affine(zb, xb, yb, out_w, out_h)
            if not (_covers(mb, out_w, out_h) and _covers(mf, out_w, out_h)):
                ok = False
                break
            if layered:
                # how far a source pixel of the foreground slides relative to
                # the background, measured in source pixels, across the frame
                inv_b = cv2.invertAffineTransform(mb)
                for px, py in ((0, 0), (out_w, out_h), (out_w / 2, out_h / 2), (0, out_h), (out_w, 0)):
                    src_f = cv2.invertAffineTransform(mf) @ np.array([px, py, 1.0])
                    src_b = inv_b @ np.array([px, py, 1.0])
                    worst = max(worst, float(np.hypot(*(src_f - src_b))))
        if ok and (not layered or worst <= REVEAL_LIMIT_PX):
            return amp, layered, {"move": move, "amplitude": amp, "layered": layered,
                                  "max_reveal_px": round(worst, 1), "fg_area": round(fg_area, 3)}
    return 0.0, False, {"move": move, "amplitude": 0.0, "layered": False, "fg_area": round(fg_area, 3),
                        "note": "no safe move: held still"}


def frames(img, bg, alpha, move, seconds, fps=30, out_w=3840, out_h=2160, fx=None, seed=0):
    """Yield RGB uint8 frames. img/bg: HxWx3 uint8 at SRC size; alpha: HxW uint8."""
    fg_area = float((alpha > 127).mean()) if alpha is not None else 0.0
    amp, layered, report = check(move, fg_area, out_w=out_w, out_h=out_h)
    if bg is None:
        layered = False
    n = max(1, int(round(seconds * fps)))
    a3 = None
    layer = None
    if fx:
        from pipeline import fx as fxmod
        layer = fxmod.Layer(fx, out_w, out_h, seed=seed)
    for k in range(n):
        t = k / max(1, n - 1)
        zf, zb, xf, xb, yf, yb = _params(move, t, amp)
        if layered:
            mb = _affine(zb, xb, yb, out_w, out_h)
            mf = _affine(zf, xf, yf, out_w, out_h)
            b = cv2.warpAffine(bg, mb, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            f = cv2.warpAffine(img, mf, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            a = cv2.warpAffine(alpha, mf, (out_w, out_h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
            out = cv2.blendLinear(f, b, a, 1.0 - a)
        else:
            m = _affine(zf, xf, yf, out_w, out_h)
            out = cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if layer is not None:
            out = layer.apply(out, k / fps)
        yield out
    frames.last_report = report


frames.last_report = None
