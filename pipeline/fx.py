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
        n = {"marine_snow": 900, "dust": 650, "embers": 380, "rain": 1400, "fog": 0}.get(kind, 0)
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
        """The particle layer as uint8 RGB at full size."""
        canvas = np.zeros((self.h, self.w), np.float32)
        pos = self.p + np.outer(self.depth, vel) * t
        pos[:, 0] += np.sin(t * 0.6 + self.phase) * 6 * self.depth
        pos[:, 0] %= self.w
        pos[:, 1] %= self.h
        bright = self.depth * (0.6 + 0.4 * np.sin(t * 3.1 + self.phase)) if flicker else self.depth
        for (x, y), sz, b in zip(pos, self.size * (0.5 + self.depth), bright):
            cv2.circle(canvas, (int(x), int(y)), max(1, int(round(sz))), float(b), -1, lineType=cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur) * alpha
        rgb = cv2.merge([canvas * c for c in color]).clip(0, 255).astype(np.uint8)
        return cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _screen(f, add):
        return cv2.subtract(255, cv2.multiply(cv2.subtract(255, f), cv2.subtract(255, add), scale=1 / 255))

    def apply(self, frame, t):
        f = frame
        if self.kind == "marine_snow":
            f = self._screen(f, self._particles(t, (4, 9), (225, 235, 240), 0.55, 0.9))
        elif self.kind == "dust":
            add = self._particles(t, (5, -3), (255, 236, 200), 1.15, 0.8)
            if not hasattr(self, "beam_rgb"):
                b = cv2.resize(self.beam, (self.W, self.H))
                self.beam_rgb = cv2.merge([b, b, b]).astype(np.float32)
                self.beam_u8 = (self.beam_rgb * 10).astype(np.uint8)
            add = cv2.multiply(add, self.beam_rgb, dtype=cv2.CV_8U)
            f = cv2.add(self._screen(f, add), self.beam_u8)
        elif self.kind == "embers":
            f = self._screen(f, self._particles(t, (6, -28), (255, 150, 60), 1.6, 0.9, flicker=True))
        elif self.kind == "rain":
            f = self._screen(f, self._particles(t, (-40, 420), (210, 220, 230), 0.35, 0.6))
        elif self.kind == "fog":
            ox, oy = int(t * 14) % self.w, int(t * 3) % self.h
            ox2 = int(t * 26) % self.w
            n1 = self.f1[oy:oy + self.h, ox:ox + self.w]
            n2 = self.f2[0:self.h, ox2:ox2 + self.w]
            fog = (0.65 * n1 + 0.35 * n2) * self.grad
            fog = np.clip((fog - 0.35) * 1.6, 0, 1) * 0.32
            fog = cv2.resize(fog, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            if not hasattr(self, "fog_col"):
                self.fog_col = np.empty((self.H, self.W, 3), np.uint8)
                self.fog_col[:] = (215, 222, 228)
            f = cv2.blendLinear(self.fog_col, f, fog, 1.0 - fog)
        return f
