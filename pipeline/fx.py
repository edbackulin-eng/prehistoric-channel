"""Animated atmosphere drawn over a 2.5D shot: fog, dust in light, marine snow,
embers, rain. Procedural, so there is nothing to license and nothing repeats.

Each layer is computed at half resolution and upscaled - particles this small
and this soft lose nothing - then screen- or alpha-blended onto the frame.
"""
import cv2
import numpy as np


def _noise(h, w, scale, rng):
    """Smooth value noise in 0..1 at (h, w), features ~`scale` px."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


class Layer:
    def __init__(self, kind, out_w, out_h, seed=0):
        self.kind, self.W, self.H = kind, out_w, out_h
        self.w, self.h = out_w // 2, out_h // 2
        self.rng = np.random.default_rng(seed + 7)
        r = self.rng
        n = {"marine_snow": 900, "dust": 500, "embers": 260, "rain": 1400, "fog": 0}.get(kind, 0)
        self.p = r.random((n, 2)) * [self.w, self.h]
        self.size = r.uniform(0.6, 2.6, n) * (self.w / 1920)
        self.depth = r.uniform(0.3, 1.0, n)
        self.phase = r.uniform(0, 2 * np.pi, n)
        if kind == "fog":
            big = (self.h * 2, self.w * 2)
            self.f1 = _noise(*big, 220, r)
            self.f2 = _noise(*big, 90, r)
            grad = np.linspace(0.35, 1.0, self.h, dtype=np.float32)[:, None]   # heavier low down
            self.grad = np.repeat(grad, self.w, axis=1)
        if kind == "dust":
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            # a shaft of light from the upper left: dust shows only inside it
            d = np.abs((xx - 0.15 * self.w) * 0.55 - (yy) * 0.83 + 0.05 * self.h) / self.w
            self.beam = np.exp(-(d / 0.16) ** 2).astype(np.float32)

    def _particles(self, t, vel, color, alpha, blur, flicker=False):
        canvas = np.zeros((self.h, self.w), np.float32)
        r = self.rng
        pos = self.p + np.outer(self.depth, vel) * t
        pos[:, 0] += np.sin(t * 0.6 + self.phase) * 6 * self.depth
        pos[:, 0] %= self.w
        pos[:, 1] %= self.h
        bright = self.depth * (0.6 + 0.4 * np.sin(t * 3.1 + self.phase)) if flicker else self.depth
        for (x, y), s, b in zip(pos, self.size * (0.5 + self.depth), bright):
            cv2.circle(canvas, (int(x), int(y)), max(1, int(round(s))), float(b), -1, lineType=cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
        a = cv2.resize(canvas, (self.W, self.H), interpolation=cv2.INTER_LINEAR) * alpha
        return a[..., None] * np.array(color, np.float32)[None, None, :]

    def apply(self, frame, t):
        f = frame.astype(np.float32)
        if self.kind == "marine_snow":
            add = self._particles(t, (4, 9), (225, 235, 240), 0.55, 0.9)
            f = 255 - (255 - f) * (1 - add / 255)                    # screen
        elif self.kind == "dust":
            add = self._particles(t, (5, -3), (255, 236, 200), 0.75, 0.8)
            beam = cv2.resize(self.beam, (self.W, self.H))[..., None]
            f = 255 - (255 - f) * (1 - add * beam / 255)
            f = f + beam * 10                                         # the shaft itself, faint
        elif self.kind == "embers":
            add = self._particles(t, (6, -28), (255, 150, 60), 0.9, 0.7, flicker=True)
            f = 255 - (255 - f) * (1 - add / 255)
        elif self.kind == "rain":
            add = self._particles(t, (-40, 420), (210, 220, 230), 0.35, 0.6)
            f = 255 - (255 - f) * (1 - add / 255)
        elif self.kind == "fog":
            ox, oy = int(t * 14) % self.w, int(t * 3) % self.h
            ox2 = int(t * 26) % self.w
            n1 = self.f1[oy:oy + self.h, ox:ox + self.w]
            n2 = self.f2[0:self.h, ox2:ox2 + self.w]
            fog = (0.65 * n1 + 0.35 * n2) * self.grad
            fog = np.clip((fog - 0.35) * 1.6, 0, 1) * 0.32
            fog = cv2.resize(fog, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[..., None]
            f = f * (1 - fog) + np.array([215, 222, 228], np.float32) * fog
        return np.clip(f, 0, 255).astype(np.uint8)
