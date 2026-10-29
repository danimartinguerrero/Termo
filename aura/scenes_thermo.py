"""
Escenas de termodinámica: gas de esferas duras que se termaliza (Maxwell–Boltzmann),
pistón con compresión adiabática, difusión / entropía, ecuación del calor
(ley cero y equilibrio), diagramas p–V de los ciclos de Carnot, Otto y
Joule–Brayton, y la superficie de estado del gas ideal pV = nRT.
"""
import math
import numpy as np
from numba import njit

from .core import (W, H, FPS, canvas, resize, blur, splat, splat_fast, fbm3, vnoise3, add_glow_point, hexcol,
                   ease_in_out, ease_out, look_at, project, rot_x, rot_y, line, polyline, polyline_fade, smoothstep)
from .scene import Scene


def _c(h, g=1.0):
    return hexcol(h, g)


# paleta térmica (frío -> caliente): azul profundo, cian, blanco, ámbar, rojo
_THERM = np.array([_c('#1b3a8a'), _c('#2fb7ff'), _c('#e9f4ff'), _c('#ffb347'), _c('#ff4b2b')], np.float32)


def thermal(x):
    """x en [0,1] -> color lineal de la paleta térmica."""
    x = np.clip(np.asarray(x, np.float32), 0, 1) * (len(_THERM) - 1)
    i = np.minimum(x.astype(int), len(_THERM) - 2)
    f = (x - i)[..., None]
    return _THERM[i] * (1 - f) + _THERM[i + 1] * f


# ───────────────────────────── gas de esferas duras ─────────────────────────────

@njit(cache=True)
def _gas_step(pos, vel, rad, x0, x1, y0, y1, vwall, dt, cell):
    n = pos.shape[0]
    for i in range(n):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        if pos[i, 0] < x0 + rad:
            pos[i, 0] = x0 + rad; vel[i, 0] = abs(vel[i, 0])
        if pos[i, 0] > x1 - rad:
            pos[i, 0] = x1 - rad
            if vel[i, 0] > vwall:
                vel[i, 0] = 2 * vwall - vel[i, 0]
        if pos[i, 1] < y0 + rad:
            pos[i, 1] = y0 + rad; vel[i, 1] = abs(vel[i, 1])
        if pos[i, 1] > y1 - rad:
            pos[i, 1] = y1 - rad; vel[i, 1] = -abs(vel[i, 1])
    # rejilla para colisiones
    gw = int((x1 - x0) / cell) + 1; gh = int((y1 - y0) / cell) + 1
    head = -np.ones(gw * gh, np.int64)
    nxt = -np.ones(n, np.int64)
    for i in range(n):
        cx = min(max(int((pos[i, 0] - x0) / cell), 0), gw - 1)
        cy = min(max(int((pos[i, 1] - y0) / cell), 0), gh - 1)
        c = cy * gw + cx
        nxt[i] = head[c]; head[c] = i
    d2min = (2 * rad) ** 2
    for i in range(n):
        cx = min(max(int((pos[i, 0] - x0) / cell), 0), gw - 1)
        cy = min(max(int((pos[i, 1] - y0) / cell), 0), gh - 1)
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                ccx = cx + ox; ccy = cy + oy
                if ccx < 0 or ccy < 0 or ccx >= gw or ccy >= gh:
                    continue
                j = head[ccy * gw + ccx]
                while j >= 0:
                    if j > i:
                        dx = pos[j, 0] - pos[i, 0]; dy = pos[j, 1] - pos[i, 1]
                        d2 = dx * dx + dy * dy
                        if d2 < d2min and d2 > 1e-9:
                            dvx = vel[j, 0] - vel[i, 0]; dvy = vel[j, 1] - vel[i, 1]
                            dot = dvx * dx + dvy * dy
                            if dot < 0:
                                k = dot / d2
                                vel[i, 0] += k * dx; vel[i, 1] += k * dy
                                vel[j, 0] -= k * dx; vel[j, 1] -= k * dy
                    j = nxt[j]


class GasParticles(Scene):
    """Gas que parte con todas las velocidades iguales y se termaliza hacia Maxwell–Boltzmann."""

    def prepare(self, dur):
        rng = self.rng
        self.box = self.get('box', (110, 330, 970, 1330))
        x0, y0, x1, y1 = self.box
        n = self.get('n', 800)
        self.rad = self.get('radius', 5.0)
        pos = np.stack([rng.uniform(x0 + 10, x1 - 10, n), rng.uniform(y0 + 10, y1 - 10, n)], 1)
        v0 = self.get('v0', 150.0)
        th = rng.random(n) * 2 * np.pi
        if self.get('init', 'mono') == 'mono':
            vel = np.stack([np.cos(th), np.sin(th)], 1) * v0
        else:
            vel = rng.normal(0, v0 / math.sqrt(2), (n, 2))
        frames = int(dur * FPS) + 2
        sub = 4
        self.P = np.zeros((frames, n, 2), np.float32)
        self.V = np.zeros((frames, n, 2), np.float32)
        self.walls = np.zeros(frames)
        piston = self.get('piston', None)    # (amplitud en px, periodo s)
        for f in range(frames):
            t = f / FPS
            if piston:
                A, Tp = piston
                xw = x1 - A * (0.5 - 0.5 * math.cos(2 * math.pi * t / Tp))
                vw = -A * math.pi / Tp * math.sin(2 * math.pi * t / Tp)
            else:
                xw, vw = x1, 0.0
            self.walls[f] = xw
            for s in range(sub):
                _gas_step(pos, vel, self.rad, x0, xw, y0, y1, vw, 1.0 / FPS / sub, 2.2 * self.rad)
            self.P[f] = pos; self.V[f] = vel
        self.v0 = v0

    def draw(self, t, u):
        img = canvas()
        f = min(int(t * FPS), self.P.shape[0] - 1)
        x0, y0, x1, y1 = self.box
        xw = self.walls[f]
        P = self.P[f]; V = self.V[f]
        sp = np.linalg.norm(V, axis=1)
        vref = self.get('vref', self.v0 * 2.0)
        cols = thermal(sp / vref) * 2.2
        # estelas de movimiento
        fp = max(f - 2, 0)
        Pp = self.P[fp]
        for i in range(0, P.shape[0], 1):
            c = cols[i] * 0.25
            if abs(P[i, 0] - Pp[i, 0]) < 40 and abs(P[i, 1] - Pp[i, 1]) < 40:
                line(img, Pp[i, 0], Pp[i, 1], P[i, 0], P[i, 1], c[0], c[1], c[2], 1.6, 0.0)
        splat(img, P[:, 0].astype(np.float32), P[:, 1].astype(np.float32), (cols * 18).astype(np.float32),
              np.full(P.shape[0], self.rad * 0.55, np.float32))
        # paredes de vidrio
        wc = _c('#a9c7ff', 0.5)
        for (a, b, c_, d) in ((x0, y0, xw, y0), (x0, y1, xw, y1), (x0, y0, x0, y1)):
            line(img, a - 6, b, c_ + 6, d, wc[0], wc[1], wc[2], 2.0, 4.0)
        if self.get('piston', None):
            pc = _c('#ffd29a', 1.0)
            line(img, xw + 6, y0 - 4, xw + 6, y1 + 4, pc[0], pc[1], pc[2], 6.0, 10.0)
            line(img, xw + 6, (y0 + y1) / 2, W + 20, (y0 + y1) / 2, pc[0] * 0.6, pc[1] * 0.6, pc[2] * 0.6, 10.0, 6.0)
        else:
            line(img, x1, y0, x1, y1, wc[0], wc[1], wc[2], 2.0, 4.0)
        if self.get('histogram', True):
            self._hist(img, sp, vref)
        return img

    def _hist(self, img, sp, vref):
        hx0, hx1, hy0, hy1 = 130, 950, 1440, 1720
        nb = 34
        vmax = vref * 1.3
        hist, edges = np.histogram(sp, bins=nb, range=(0, vmax))
        dens = hist / (sp.size * (vmax / nb))
        s2 = np.mean(sp ** 2) / 2
        vv = np.linspace(0, vmax, 200)
        mb = vv / s2 * np.exp(-vv ** 2 / (2 * s2))
        top = max(mb.max(), 1e-9) * 1.15
        bw = (hx1 - hx0) / nb
        for k in range(nb):
            hgt = min(dens[k] / top, 1.25) * (hy1 - hy0)
            c = thermal((edges[k] + edges[k + 1]) / 2 / vref) * 0.9
            xa = hx0 + k * bw + 2
            line(img, xa + bw / 2 - 1, hy1, xa + bw / 2 - 1, hy1 - hgt, c[0], c[1], c[2], bw - 5, 0.0)
        ax = _c('#e8eef8', 0.5)
        line(img, hx0, hy1, hx1, hy1, ax[0], ax[1], ax[2], 1.5, 0.0)
        pts = np.stack([hx0 + vv / vmax * (hx1 - hx0), hy1 - mb / top * (hy1 - hy0)], 1).astype(np.float32)
        gc = _c('#ffd27a', 1.6)
        polyline(img, pts, gc[0], gc[1], gc[2], 2.6, 6.0)


# ───────────────────────────── difusión / entropía ─────────────────────────────

@njit(cache=True)
def _advect(px, py, t, dt, scale, strength, seed, jitter, rnd):
    e = 0.01
    for i in range(px.shape[0]):
        x = px[i] * scale; y = py[i] * scale
        # campo de rizo (curl noise): divergencia nula -> filamentos tipo tinta
        n1 = fbm3(x, y + e, t * 0.05, 4, seed); n2 = fbm3(x, y - e, t * 0.05, 4, seed)
        n3 = fbm3(x + e, y, t * 0.05, 4, seed); n4 = fbm3(x - e, y, t * 0.05, 4, seed)
        vx = (n1 - n2) / (2 * e); vy = -(n3 - n4) / (2 * e)
        px[i] += vx * strength * dt + rnd[i, 0] * jitter
        py[i] += vy * strength * dt + rnd[i, 1] * jitter
        if px[i] < 0: px[i] = -px[i]
        if px[i] > 1: px[i] = 2 - px[i]
        if py[i] < 0: py[i] = -py[i]
        if py[i] > 1: py[i] = 2 - py[i]


class InkEntropy(Scene):
    """Dos poblaciones (dorada y cian) empiezan separadas y se mezclan: la flecha del tiempo."""

    def prepare(self, dur):
        rng = self.rng
        n = self.get('n', 240000)
        mode = self.get('mode', 'drop')
        if mode == 'drop':
            r = np.sqrt(rng.random(n)) * 0.06; th = rng.random(n) * 2 * np.pi
            px = 0.5 + r * np.cos(th) * (H / W); py = 0.45 + r * np.sin(th)
            kind = (rng.random(n) < 0.5)
        else:
            px = rng.random(n); py = rng.random(n)
            kind = px < 0.5
            px = np.where(kind, px * 0.98, 0.02 + px * 0.98)
        frames = int(dur * FPS) + 2
        self.frames = np.zeros((frames, n, 2), np.float32)
        px = px.astype(np.float64); py = py.astype(np.float64)
        for f in range(frames):
            rnd = rng.normal(0, 1, (n, 2))
            _advect(px, py, f / FPS, 1 / FPS, 2.2, self.get('flow', 0.12), self.seed, self.get('jitter', 0.0028), rnd)
            self.frames[f, :, 0] = px; self.frames[f, :, 1] = py
        ca = _c(self.get('color_a', '#ffb347'), 1.0); cb = _c(self.get('color_b', '#36c8ff'), 1.0)
        self.cols = np.where(kind[:, None], ca, cb).astype(np.float32) * self.get('gain', 0.035)

    def draw(self, t, u):
        f = min(int(t * FPS), self.frames.shape[0] - 1)
        img = canvas()
        P = self.frames[f]
        m0 = 0.06
        x = (m0 + P[:, 0] * (1 - 2 * m0)) * W
        y = (0.12 + P[:, 1] * 0.76) * H
        splat_fast(img, x, y, self.cols)
        img = img * 0.7 + blur(img, 1.6) * 1.2 + blur(img, 8) * 0.6
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]; xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        img *= (smoothstep(0.1, 0.18, yy) * smoothstep(0.9, 0.82, yy) * smoothstep(0.04, 0.1, xx) * smoothstep(0.96, 0.9, xx))[..., None]
        return img


# ───────────────────────────── ecuación del calor ─────────────────────────────

class HeatDiffusion(Scene):
    """Mapa de temperatura que evoluciona hacia el equilibrio térmico, con isotermas."""
    scale = 0.25

    def prepare(self, dur):
        w, h = self.size()
        rng = self.rng
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        T = np.zeros((h, w))
        for k in range(self.get('sources', 16)):
            cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
            s = rng.uniform(0.06, 0.16) * w
            T += rng.choice([1.0, -0.7]) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        T = (T - T.min()) / (T.max() - T.min())
        frames = int(dur * FPS) + 2
        self.F = np.zeros((frames, h, w), np.float32)
        D = self.get('diffusivity', 0.22)
        for f in range(frames):
            for s in range(self.get('substeps', 6)):
                lap = (np.roll(T, 1, 0) + np.roll(T, -1, 0) + np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T)
                T = T + D * lap
            self.F[f] = T
        self.mean = float(T.mean())

    def draw(self, t, u):
        f = min(int(t * FPS), self.F.shape[0] - 1)
        T = resize(self.F[f], W, H)
        T = np.clip(T, 0, 1)
        img = _inferno(T) * 0.9
        lv = self.get('levels', 14)
        fr = (T * lv) % 1.0
        edge = np.abs(fr - 0.5)
        iso = np.clip(1 - (0.5 - edge) / 0.03, 0, 1)
        img += iso[..., None] * _c('#ffffff', 0.25)
        return img


_INF = np.array([_c('#000004'), _c('#1b0c41'), _c('#4a0c6b'), _c('#781c6d'), _c('#a52c60'),
                 _c('#cf4446'), _c('#ed6925'), _c('#fb9b06'), _c('#f7d13d'), _c('#fcffa4')], np.float32)


def _inferno(x):
    x = np.clip(x, 0, 1) * (len(_INF) - 1)
    i = np.minimum(x.astype(int), len(_INF) - 2)
    f = (x - i)[..., None]
    return _INF[i] * (1 - f) + _INF[i + 1] * f


# ───────────────────────────── diagramas p–V ─────────────────────────────

class PVCycle(Scene):
    """Diagrama p–V con isotermas y adiabáticas de fondo y un ciclo que se traza con luz."""

    def prepare(self, dur):
        self.gamma = self.get('gamma', 1.4)
        kind = self.get('cycle', 'carnot')
        g = self.gamma
        n = 120
        if kind == 'carnot':
            Va, Vb = 1.0, 2.2
            Th, Tc = 4.0, 2.6
            Vc = Vb * (Th / Tc) ** (1 / (g - 1)); Vd = Va * (Th / Tc) ** (1 / (g - 1))
            seg = [('iso', Va, Vb, Th), ('adi', Vb, Vc, Th * Vb ** (g - 1)), ('iso', Vc, Vd, Tc), ('adi', Vd, Va, Tc * Vd ** (g - 1))]
        elif kind == 'otto':
            V1, V2 = 4.2, 1.0
            p1 = 0.9
            K1 = p1 * V1 ** g
            p2 = K1 / V2 ** g
            p3 = p2 * 1.9
            K2 = p3 * V2 ** g
            seg = [('adiK', V1, V2, K1), ('isoV', V2, p2, p3), ('adiK', V2, V1, K2), ('isoV', V1, K2 / V1 ** g, p1)]
        else:   # brayton
            p_lo, p_hi = 0.8, 3.4
            V1 = 3.0
            V2 = V1 * (p_lo / p_hi) ** (1 / g)
            V3 = 1.9
            V4 = V3 * (p_hi / p_lo) ** (1 / g)
            seg = [('adiK', V1, V2, p_lo * V1 ** g), ('isoP', V2, V3, p_hi), ('adiK', V3, V4, p_hi * V3 ** g), ('isoP', V4, V1, p_lo)]
        pts = []
        for s in seg:
            tt = np.linspace(0, 1, n)
            if s[0] == 'iso':
                V = s[1] + (s[2] - s[1]) * tt; p = s[3] / V
            elif s[0] == 'adi':
                V = s[1] + (s[2] - s[1]) * tt; p = s[3] / V ** g
            elif s[0] == 'adiK':
                V = s[1] + (s[2] - s[1]) * tt; p = s[3] / V ** g
            elif s[0] == 'isoV':
                V = np.full(n, s[1]); p = s[2] + (s[3] - s[2]) * tt
            else:
                V = s[1] + (s[2] - s[1]) * tt; p = np.full(n, s[3])
            pts.append(np.stack([V, p], 1))
        self.cyc = np.concatenate(pts)
        self.vmax = self.cyc[:, 0].max() * 1.18
        self.pmax = self.cyc[:, 1].max() * 1.18
        self.T = self.cyc[:, 0] * self.cyc[:, 1]

    def to_px(self, V, p):
        x0, y0, x1, y1 = 140, 420, 960, 1420
        return x0 + V / self.vmax * (x1 - x0), y1 - p / self.pmax * (y1 - y0)

    def draw(self, t, u):
        img = canvas()
        rev = ease_out(min(u * 3, 1.0))
        x0, y0, x1, y1 = 140, 420, 960, 1420
        ax = _c('#e8eef8', 0.8 * rev)
        line(img, x0, y1, x0 + (x1 - x0 + 30) * rev, y1, ax[0], ax[1], ax[2], 2.0, 2.0)
        line(img, x0, y1, x0, y1 - (y1 - y0 + 40) * rev, ax[0], ax[1], ax[2], 2.0, 2.0)
        V = np.linspace(0.25, self.vmax, 300)
        # familia de isotermas (azul) y adiabáticas (ámbar)
        for k, c in enumerate(np.linspace(0.6, 5.0, 9)):
            p = c / V
            m = p < self.pmax
            X, Y = self.to_px(V[m], p[m])
            pts = np.stack([X, Y], 1).astype(np.float32)
            cc = _c('#4f7dff', 0.18 * rev)
            if len(pts) > 1:
                polyline(img, pts, cc[0], cc[1], cc[2], 1.2, 0.0)
        for c in np.linspace(0.6, 8.0, 8):
            p = c / V ** self.gamma
            m = p < self.pmax
            X, Y = self.to_px(V[m], p[m])
            pts = np.stack([X, Y], 1).astype(np.float32)
            cc = _c('#ff9a3c', 0.13 * rev)
            if len(pts) > 1:
                polyline(img, pts, cc[0], cc[1], cc[2], 1.2, 0.0)
        # trazado del ciclo
        draw_u = np.clip((u - 0.12) / self.get('trace', 0.6), 0, 1)
        N = len(self.cyc)
        k = int(draw_u * (N - 1))
        if k > 1:
            X, Y = self.to_px(self.cyc[:k, 0], self.cyc[:k, 1])
            Tn = (self.T[:k] - self.T.min()) / (self.T.max() - self.T.min())
            cols = (0.15 + 0.85 * Tn)
            for i in range(k - 1):
                c = _c('#3fa9ff') * (1 - cols[i]) + _c('#ff7a2f') * cols[i]
                c = c * 1.3
                line(img, X[i], Y[i], X[i + 1], Y[i + 1], c[0], c[1], c[2], 3.0, 6.0)
            hx, hy = X[-1], Y[-1]
            add_glow_point(img, hx, hy, _c('#ffffff', 1.2), 5.0)
        # área encerrada = trabajo neto
        if draw_u >= 1.0:
            fa = ease_in_out((u - 0.12 - self.get('trace', 0.6)) / 0.15) * 0.18
            X, Y = self.to_px(self.cyc[:, 0], self.cyc[:, 1])
            import cv2
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [np.stack([X, Y], 1).astype(np.int32)], 1)
            img += (mask[..., None] * fa).astype(np.float32) * _c('#ffc66b')
        return img


# ───────────────────────────── superficie pV = nRT ─────────────────────────────

class IdealGasSurface(Scene):
    def draw(self, t, u):
        img = canvas()
        ang = -0.6 + 0.25 * t
        cam = look_at((math.sin(ang) * 11, 5.5, -math.cos(ang) * 11), (0, 1.3, 0))
        fov = 45

        def P(V, T):
            p = T / V
            return np.stack([(V - 2.2) * 1.3, np.minimum(p, 3.2) * 1.0, (T - 2.0) * 1.3], -1)

        Vs = np.linspace(0.6, 4.0, 160)
        Ts = np.linspace(0.6, 3.4, 160)
        rev = ease_out(min(u * 2, 1.0))
        for T in np.linspace(0.6, 3.4, 15):
            Q = P(Vs, np.full_like(Vs, T))
            x, y, z = project(Q, cam, fov)
            pts = np.stack([x, y], 1).astype(np.float32)
            c = _c('#3fa9ff') * (1 - (T - 0.6) / 2.8) + _c('#ff7a2f') * ((T - 0.6) / 2.8)
            polyline(img, pts, *(c * 1.6 * rev), 1.8, 5.0)
        for V in np.linspace(0.6, 4.0, 15):
            Q = P(np.full_like(Ts, V), Ts)
            x, y, z = project(Q, cam, fov)
            pts = np.stack([x, y], 1).astype(np.float32)
            polyline(img, pts, *(_c('#9fb6d9') * 0.35 * rev), 1.0, 0.0)
        # un estado recorriendo la superficie
        s = t * 0.5
        V = 2.0 + 1.2 * math.sin(s); T = 2.0 + 1.0 * math.sin(s * 1.3 + 1)
        x, y, z = project(P(np.array([V]), np.array([T])), cam, fov)
        add_glow_point(img, x[0], y[0], _c('#ffffff', 1.5), 6.0)
        return img
