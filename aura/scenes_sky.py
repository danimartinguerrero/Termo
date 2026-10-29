"""
Escenas de cielo y cosmos: campos de estrellas, Vía Láctea con paisaje,
nebulosas tipo JWST, galaxia espiral, auroras, la Tierra desde órbita.
"""
import math
import numpy as np
from numba import njit, prange

from .core import (W, H, canvas, resize, blur, splat, splat_fast, blackbody, fbm3, ridged3, vnoise3,
                   add_glow_point, hexcol, ease_in_out, look_at, project, rot_x, rot_y, rot_z, line, polyline)
from .scene import Scene


# ───────────────────────────── estrellas compartidas ─────────────────────────────

class StarLayer:
    """Nube de estrellas en la esfera celeste, con brillo según ley de potencias."""

    def __init__(self, n, rng, mag_scale=1.0, band=None):
        self.n = n
        if band is None:
            v = rng.normal(size=(n, 3))
        else:  # concentrar parte en un plano galáctico
            v = rng.normal(size=(n, 3))
            m = rng.random(n) < band
            v[m, 1] *= 0.12
        self.dir = v / np.linalg.norm(v, axis=1, keepdims=True)
        T = np.clip(rng.lognormal(math.log(6000), 0.45, n), 2500, 25000)
        self.col = blackbody(T)
        self.col /= self.col.max(axis=1, keepdims=True)
        self.lum = (rng.pareto(1.6, n) + 1.0) * 0.6 * mag_scale
        self.lum = np.minimum(self.lum, 60.0)
        self.phase = rng.random(n) * 6.283
        self.freq = rng.uniform(0.5, 3.0, n)

    def draw(self, img, cam, fov, t, twinkle=0.15, gain=1.0, size=0.8, bright_glow=True, alive=None):
        h, w, _ = img.shape
        x, y, z = project(self.dir * 1000.0 + cam[0], cam, fov, w, h)
        m = (z > 0) & (x > -10) & (x < w + 10) & (y > -10) & (y < h + 10)
        if alive is not None:
            m &= alive > 0.002
        lum = self.lum * (1 + twinkle * np.sin(self.phase + t * self.freq * 6.283)) * gain
        if alive is not None:
            lum = lum * alive
        idx = np.nonzero(m)[0]
        cols = (self.col[idx] * lum[idx, None]).astype(np.float32)
        sizes = np.full(idx.size, size, np.float32) * (1 + 0.15 * np.log1p(lum[idx]))
        splat(img, x[idx].astype(np.float32), y[idx].astype(np.float32), cols, sizes.astype(np.float32))
        if bright_glow:
            b = idx[lum[idx] > 14 * gain]
            for i in b[:40]:
                add_glow_point(img, x[i], y[i], self.col[i] * 0.04 * lum[i], 2.5 * w / W)
        return img


# ───────────────────────────── Vía Láctea + paisaje ─────────────────────────────

@njit(parallel=True, cache=True)
def _milkyway(out, t, ox, oy, ang, width, seed, core_x, core_y):
    h, w, _ = out.shape
    ca = math.cos(ang); sa = math.sin(ang)
    for j in prange(h):
        for i in range(w):
            x = (i / w - 0.5) * (w / h) + ox
            y = (j / h - 0.5) + oy
            u = x * ca + y * sa          # a lo largo de la banda
            v = -x * sa + y * ca         # perpendicular
            wv = v + 0.03 * (fbm3(u * 2.0, 0.0, 0.5, 3, seed + 2) - 0.5)
            band = math.exp(-(wv * wv) / (2 * width * width))
            dcore = math.sqrt((u - core_x) ** 2 + (wv - core_y) ** 2 * 4)
            core = math.exp(-dcore * 3.0)
            n = fbm3(u * 4.0, v * 8.0, 0.3, 7, seed)
            clump = fbm3(u * 30.0, v * 30.0, 1.7, 3, seed + 5)
            dust = fbm3(u * 6.0 + 3.0, v * 14.0, 2.1, 6, seed + 9)
            dust2 = ridged3(u * 9.0, v * 20.0, 4.1, 5, seed + 19)
            dl = math.exp(-((wv + 0.01 * math.sin(u * 9)) ** 2) / (2 * (width * 0.45) ** 2))
            ext = min(max((dust - 0.42) * 3.5, 0.0), 1.0) * dl + 0.5 * dust2 * dust2 * dl
            ext = min(ext, 1.0)
            glow = band * band * (0.15 + 0.85 * n * n * 1.6) * (0.6 + 0.7 * clump) * (0.3 + 1.6 * core)
            glow *= (1.0 - 0.95 * ext)
            warm = min(core * 1.4, 1.0)
            r = glow * (0.42 + 0.6 * warm); g = glow * (0.45 + 0.33 * warm); b = glow * (0.7 - 0.3 * warm)
            hii = fbm3(u * 34.0, v * 34.0, 4.0, 4, seed + 13)
            hii = max(hii - 0.72, 0.0) * 3.0 * band * (1.0 - ext)
            out[j, i, 0] = r * 0.32 + hii * 0.25
            out[j, i, 1] = g * 0.32 + hii * 0.04
            out[j, i, 2] = b * 0.32 + hii * 0.09


@njit(cache=True)
def _ridge_line(xs, seed, scale, amp, base):
    out = np.empty(xs.shape[0])
    for i in range(xs.shape[0]):
        n = 0.0; a = 1.0; f = scale; norm = 0.0
        for o in range(6):
            n += a * vnoise3(xs[i] * f, 0.5, 0.5, seed + o)
            norm += a; a *= 0.5; f *= 2.1
        out[i] = base - amp * n / norm
    return out


class MilkyWay(Scene):
    """Cielo nocturno con Vía Láctea, horizonte montañoso y silueta humana opcional."""
    scale = 0.5

    def prepare(self, dur):
        self.stars = StarLayer(26000, self.rng, 0.9, band=0.5)
        w, h = W, H
        xs = np.arange(w, dtype=np.float64) / w
        self.horizon_y = self.get('horizon', 0.84)
        self.ridge = _ridge_line(xs, self.seed, 3.0, 0.08 * h, self.horizon_y * h) if self.horizon_y < 1 else None
        self.ridge2 = _ridge_line(xs, self.seed + 40, 2.0, 0.05 * h, (self.horizon_y + 0.04) * h) if self.horizon_y < 1 else None

    def draw(self, t, u):
        w, h = self.size()
        sky = np.zeros((h, w, 3), np.float32)
        drift = self.get('drift', 0.04)
        ang = self.get('angle', -1.05) + t * self.get('spin', 0.004)
        _milkyway(sky, t, 0.1 + drift * u, -0.05, ang, self.get('width', 0.09), self.seed, 0.15, 0.0)
        img = resize(sky, W, H) * self.get('mw_gain', 1.0)
        self._band_stars(img, 0.1 + drift * u, -0.05, ang, self.get('width', 0.09))
        # rotación lenta del cielo
        yaw = 0.3 + t * self.get('spin', 0.004)
        cam = look_at((0, 0, 0), (math.sin(yaw), 0.35 + 0.02 * u, math.cos(yaw)), (0, 1, 0))
        self.stars.draw(img, cam, self.get('fov', 70), t, gain=self.get('star_gain', 1.0))
        # resplandor del horizonte (airglow / contaminación tenue)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        hz = self.horizon_y
        glow = np.exp(-np.maximum(hz - yy, 0) / 0.08) * (yy < hz + 0.1)
        img += glow * hexcol(self.get('horizon_color', '#2a3b5c'), 0.35)
        if self.ridge is not None:
            yi = np.arange(H, dtype=np.float32)[:, None]
            far = yi > self.ridge[None, :]
            near = yi > self.ridge2[None, :]
            img[far] = img[far] * 0.1 + hexcol('#1a2238', 0.3)
            img[near] = hexcol('#020306', 0.2)
            if self.get('figure', False):
                self._figure(img)
        return img

    def _band_stars(self, img, ox, oy, ang, width):
        if not hasattr(self, 'bu'):
            n = 90000
            self.bu = self.rng.uniform(-1.2, 1.2, n)
            self.bv = self.rng.normal(0, 1, n) * width * 0.8
            T = np.clip(self.rng.lognormal(math.log(5500), 0.35, n), 2500, 20000)
            c = blackbody(T); self.bc = c / c.max(1, keepdims=True) * (self.rng.pareto(2.5, n)[:, None] * 0.2 + 0.12)
        ca, sa = math.cos(ang), math.sin(ang)
        x = self.bu * ca - self.bv * sa - ox
        y = self.bu * sa + self.bv * ca - oy
        X = (x / (W / H) + 0.5) * W; Y = (y + 0.5) * H
        splat_fast(img, X.astype(np.float32), Y.astype(np.float32), self.bc.astype(np.float32))

    def _figure(self, img):
        fx = int(W * self.get('figure_x', 0.62))
        base = int(self.ridge2[fx]) + 2
        hgt = int(H * self.get('figure_h', 0.075))
        yy, xx = np.mgrid[base - hgt - 30:base + 2, fx - hgt:fx + hgt]
        dx = (xx - fx) / hgt; dy = (base - yy) / hgt
        head = (dx ** 2 + (dy - 0.9) ** 2) < 0.012
        body = (dy > 0) & (dy < 0.82) & (np.abs(dx) < 0.09 + 0.05 * np.clip(dy - 0.45, 0, 1) * (dy < 0.78))
        legs = (dy >= 0) & (dy < 0.42) & (np.abs(np.abs(dx) - 0.035) < 0.03)
        m = head | body | legs
        img[yy[m], xx[m]] = 0.003


# ───────────────────────────── Nebulosa (estilo JWST) ─────────────────────────────

PALETTES = {
    # acantilados cósmicos: gas azul encima, polvo anaranjado abajo
    'cliffs': dict(gas=('#1e4f9c', '#7fc8ff'), dust=('#2a0f05', '#c8662a', '#ffcf8a'), bg='#02040a'),
    # pilares de la creación (JWST NIRCam): rojizos y dorados
    'pillars': dict(gas=('#5a1630', '#ff9a6b'), dust=('#140805', '#8a4a22', '#f2c27a'), bg='#05030a'),
    # nebulosa de emisión oxígeno/hidrógeno
    'teal': dict(gas=('#0b3a46', '#6ff2e0'), dust=('#1a0410', '#b0305a', '#ff9fb8'), bg='#010307'),
    # violeta frío
    'violet': dict(gas=('#2a1660', '#b39bff'), dust=('#0a0614', '#5a2a7a', '#f5b6ff'), bg='#020108'),
}


@njit(parallel=True, cache=True)
def _nebula(out, t, zoom, ox, oy, seed, g0, g1, d0, d1, d2, bg, dust_par, slope, evolve):
    h, w, _ = out.shape
    asp = w / h
    for j in prange(h):
        for i in range(w):
            x0 = ((i / w - 0.5) * asp) / zoom + ox
            y0 = (j / h - 0.5) / zoom + oy
            z = t * evolve
            # capa de gas: warping de dominio
            qx = fbm3(x0 * 1.6, y0 * 1.6, z, 5, seed)
            qy = fbm3(x0 * 1.6 + 5.2, y0 * 1.6 + 1.3, z, 5, seed + 3)
            gas = fbm3(x0 * 2.2 + 3.0 * qx, y0 * 2.2 + 3.0 * qy, z * 0.5, 6, seed + 7)
            fil = ridged3(x0 * 3.0 + 2.0 * qx, y0 * 3.0 + 2.0 * qy, z, 5, seed + 11)
            gas = max(gas - 0.3, 0.0) * 1.6 + fil * fil * 0.45
            # capa de polvo con paralaje
            xd = ((i / w - 0.5) * asp) / (zoom * dust_par) + ox * dust_par
            yd = (j / h - 0.5) / (zoom * dust_par) + oy * dust_par
            px = fbm3(xd * 1.3, yd * 1.3, z + 7.0, 5, seed + 21)
            dden = fbm3(xd * 2.0 + 2.5 * px, yd * 2.0, z * 0.3, 7, seed + 25)
            # frontera inclinada (acantilado): más polvo en la parte baja
            edge = (yd + slope * xd) * 2.2 + (dden - 0.5) * 2.8
            dust = min(max(edge * 2.5 + 0.3, 0.0), 1.0)
            deep = min(max(edge, 0.0), 1.5)
            fine = fbm3(xd * 9.0 + px, yd * 9.0, z, 4, seed + 33)
            # iluminación del borde del polvo (frente de ionización)
            rim = math.exp(-((edge + 0.02 * (fine - 0.5)) ** 2) / 0.006) * (0.3 + fine * fine * 1.6)
            gas += math.exp(-((edge + 0.12) ** 2) / 0.03) * 0.5 * (0.4 + fine)
            # composición
            r = bg[0] + gas * (g0[0] + (g1[0] - g0[0]) * min(gas, 1.0))
            g = bg[1] + gas * (g0[1] + (g1[1] - g0[1]) * min(gas, 1.0))
            b = bg[2] + gas * (g0[2] + (g1[2] - g0[2]) * min(gas, 1.0))
            a = min(dust, 1.0)
            tex = (0.15 + 1.6 * dden * dden * fine * 1.8) * math.exp(-deep * 1.6)
            dr = d0[0] + (d1[0] - d0[0]) * tex; dg = d0[1] + (d1[1] - d0[1]) * tex; db = d0[2] + (d1[2] - d0[2]) * tex
            r = r * (1 - a) + dr * a + d2[0] * rim * 0.45
            g = g * (1 - a) + dg * a + d2[1] * rim * 0.45
            b = b * (1 - a) + db * a + d2[2] * rim * 0.45
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b


class Nebula(Scene):
    scale = 0.5

    def prepare(self, dur):
        pal = PALETTES[self.get('palette', 'cliffs')]
        self.g0, self.g1 = [hexcol(c, 1.0) for c in pal['gas']]
        self.d0, self.d1, self.d2 = [hexcol(c, 1.0) for c in pal['dust']]
        self.bg = hexcol(pal['bg'], 1.0)
        n = 9000
        self.sx = self.rng.random(n); self.sy = self.rng.random(n)
        T = np.clip(self.rng.lognormal(math.log(7000), 0.5, n), 3000, 30000)
        self.sc = blackbody(T); self.sc /= self.sc.max(1, keepdims=True)
        self.sl = np.minimum(self.rng.pareto(1.4, n) + 1, 80) * 0.45
        self.sdepth = self.rng.uniform(0.6, 1.6, n)

    def draw(self, t, u):
        w, h = self.size()
        neb = np.zeros((h, w, 3), np.float32)
        z0, z1 = self.get('zoom', (1.0, 1.25))
        zoom = z0 * (z1 / z0) ** ease_in_out(u) if self.get('ease', True) else z0 * (z1 / z0) ** u
        ox, oy = self.get('offset', (0.0, 0.0))
        pan = self.get('pan', (0.02, -0.01))
        ox += pan[0] * u; oy += pan[1] * u
        _nebula(neb, t, zoom, ox, oy, self.seed, self.g0, self.g1, self.d0, self.d1, self.d2, self.bg,
                self.get('dust_parallax', 0.8), self.get('slope', 0.25), self.get('evolve', 0.02))
        img = resize(neb, W, H) * self.get('gain', 1.0)
        # estrellas con paralaje y picos de difracción JWST
        cx, cy = 0.5, 0.5
        zz = zoom ** self.sdepth
        x = ((self.sx - 0.5 - (ox * 0.0)) * zz + cx) * W
        y = ((self.sy - 0.5) * zz + cy) * H
        m = (x > -50) & (x < W + 50) & (y > -50) & (y < H + 50)
        idx = np.nonzero(m)[0]
        tw = 1 + 0.08 * np.sin(t * 2 + idx)
        splat(img, x[idx].astype(np.float32), y[idx].astype(np.float32),
              (self.sc[idx] * (self.sl[idx] * tw)[:, None]).astype(np.float32),
              np.full(idx.size, 0.9, np.float32))
        big = idx[self.sl[idx] > 16][:14]
        for i in big:
            L = self.sl[i]
            add_glow_point(img, x[i], y[i], self.sc[i] * 0.03 * L, 2.5, spikes=8,
                           spike_len=min(9 * L ** 0.65, 170), angle=math.pi / 2, spike_w=1.0)
        return img


# ───────────────────────────── Galaxia espiral ─────────────────────────────

class SpiralGalaxy(Scene):
    """Galaxia espiral de gran diseño: disco, brazos logarítmicos, polvo y bulbo."""

    def prepare(self, dur):
        rng = self.rng
        pitch = math.radians(self.get('pitch', 14))
        k = 1 / math.tan(pitch)

        def arm_pts(n, spread, r_scale, off=0.0):
            r = rng.exponential(r_scale, n) + 0.08
            r = r[r < 1.5]
            arm = rng.integers(0, self.get('arms', 2), r.size)
            th = np.log(r / 0.1) * k * 0.5 + arm * 2 * np.pi / self.get('arms', 2) + off
            th = th + rng.normal(0, spread, r.size) * (0.6 + 0.6 * r)
            return r, th

        # disco difuso
        nd = 220000
        rd = rng.exponential(0.33, nd); rd = rd[rd < 1.6]
        td = rng.random(rd.size) * 2 * np.pi
        # brazos (estrellas jóvenes)
        ra, ta = arm_pts(200000, 0.3, 0.55)
        # regiones HII
        rh, th_ = arm_pts(2500, 0.12, 0.45)
        # bulbo
        nb = 60000
        bul = rng.normal(0, 1, (nb, 3)) * np.array([0.09, 0.06, 0.09])
        self.r = np.concatenate([rd, ra, rh]); self.th = np.concatenate([td, ta, th_])
        self.zs = np.concatenate([rng.normal(0, 0.02, rd.size), rng.normal(0, 0.012, ra.size), rng.normal(0, 0.008, rh.size)])
        self.bulge = bul
        Td = np.clip(rng.normal(5200, 900, rd.size), 3000, 9000)
        Ta = np.clip(rng.normal(11000, 4000, ra.size), 5000, 30000)
        cols = np.concatenate([blackbody(Td), blackbody(Ta), np.tile(hexcol('#ff4f86', 1.0), (rh.size, 1))])
        cols /= cols.max(1, keepdims=True)
        lum = np.concatenate([rng.pareto(2.5, rd.size) * 0.03 + 0.025 * np.exp(-rd / 0.5) + 0.01,
                              rng.pareto(2.0, ra.size) * 0.06 + 0.03,
                              rng.pareto(2.0, rh.size) * 0.4 + 0.3])
        self.cols = cols * lum[:, None]
        Tb = np.clip(rng.normal(4300, 400, nb), 3000, 6000)
        cb = blackbody(Tb); self.bcols = cb / cb.max(1, keepdims=True) * 0.05
        # polvo, en la cara interna de los brazos
        rdu, tdu = arm_pts(90000, 0.16, 0.55, off=-0.3)
        self.rdu, self.tdu = rdu, tdu
        self.zdu = rng.normal(0, 0.006, rdu.size)
        self.bg = StarLayer(9000, rng, 0.5)

    @staticmethod
    def _xyz(r, th, z, rot):
        omega = rot / (0.25 + r)
        a = th + omega
        return np.stack([r * np.cos(a), z, r * np.sin(a)], 1)

    def draw(self, t, u):
        img = canvas()
        rot = t * self.get('speed', 0.05)
        tilt = self.get('tilt', (1.05, 0.8))
        tl = tilt[0] + (tilt[1] - tilt[0]) * ease_in_out(u)
        R = rot_x(tl) @ rot_z(self.get('roll', 0.35))
        dist = self.get('dist', (3.0, 2.3))
        d = dist[0] + (dist[1] - dist[0]) * ease_in_out(u)
        cam = look_at((0, 0, -d), (0, 0, 0), (0, 1, 0))
        fov = 45
        self.bg.draw(img, cam, fov, t, gain=0.6, bright_glow=False)
        P = self._xyz(self.r, self.th, self.zs, rot) @ R.T
        x, y, z = project(P, cam, fov)
        lay = np.zeros_like(img)
        splat_fast(lay, x.astype(np.float32), y.astype(np.float32), (self.cols * (6 / d ** 2)).astype(np.float32))
        Pb = self.bulge @ R.T
        xb, yb, zb = project(Pb, cam, fov)
        splat_fast(lay, xb.astype(np.float32), yb.astype(np.float32), (self.bcols * (6 / d ** 2)).astype(np.float32))
        # polvo
        Pd = self._xyz(self.rdu, self.tdu, self.zdu, rot) @ R.T
        xd, yd, zd = project(Pd, cam, fov)
        dust = np.zeros((H, W, 3), np.float32)
        splat_fast(dust, xd.astype(np.float32), yd.astype(np.float32), np.full((xd.size, 3), 1.0, np.float32))
        dust = blur(dust[..., :1].copy(), 5.0)
        ext = np.exp(-dust * 1.1 * (2.3 / d) ** 2)
        lay = lay * 0.9 + blur(lay, 2.0) * 0.6 + blur(lay, 9) * 0.9 + blur(lay, 30) * 0.8
        lay *= ext[..., None] if ext.ndim == 2 else ext
        img += lay
        cx, cy, _ = project(np.zeros((1, 3)), cam, fov)
        add_glow_point(img, cx[0], cy[0], hexcol('#ffe2b8', 0.5), 12 / d)
        return img


# ───────────────────────────── Auroras desde el suelo ─────────────────────────────

@njit(parallel=True, cache=True)
def _aurora(out, t, seed, horizon, layers, speed, hue_shift):
    h, w, _ = out.shape
    asp = w / h
    for j in prange(h):
        yv = j / h
        if yv > horizon + 0.01:
            continue
        for i in range(w):
            xv = (i / w - 0.5) * asp
            r = 0.0; g = 0.0; b = 0.0
            for k in range(layers):
                depth = 1.0 + k * 0.6
                xs = xv * 1.2 * depth + k * 3.7
                # línea base ondulada (pliegues de la cortina)
                fold = fbm3(xs * 2.4 + t * speed * 0.05, t * speed * 0.12 + k * 1.3, 0.0, 4, seed + k)
                base = horizon - 0.1 - 0.2 / depth + 0.3 * (fold - 0.5) / depth
                d = base - yv                      # >0 por encima de la base
                if d < -0.03:
                    continue
                ray = fbm3(xs * 34.0 + 6.0 * fold, t * speed * 0.5, k * 2.0, 3, seed + 40)
                ray = 0.12 + 2.2 * ray * ray * ray
                tall = (0.05 + 0.16 * fbm3(xs * 1.7, t * 0.08, k + 3.0, 3, seed + 60)) / math.sqrt(depth)
                if d >= 0:
                    fall = math.exp(-d / tall) * (1.0 - math.exp(-d / 0.004))
                    fall += 0.9 * math.exp(-d / 0.012)
                else:
                    fall = math.exp(d / 0.0035) * 0.9
                inten = fall * ray / depth ** 1.1
                inten *= 0.05 + 3.0 * fbm3(xs * 0.7, t * speed * 0.1, k * 5.0, 3, seed + 80) ** 3
                top = min(max(d, 0.0) / 0.28, 1.0)
                r += inten * (0.04 + 0.6 * top * top + hue_shift * 0.3)
                g += inten * (1.0 - 0.8 * top)
                b += inten * (0.25 + 0.42 * top)
            out[j, i, 0] += r * 0.5; out[j, i, 1] += g * 0.5; out[j, i, 2] += b * 0.5


class AuroraSky(Scene):
    scale = 0.5

    def prepare(self, dur):
        self.stars = StarLayer(14000, self.rng, 0.7)
        xs = np.arange(W, dtype=np.float64) / W
        self.horizon = self.get('horizon', 0.7)
        self.ridge = _ridge_line(xs, self.seed + 3, 2.4, 0.07 * H, self.horizon * H)

    def draw(self, t, u):
        w, h = self.size()
        sky = np.zeros((h, w, 3), np.float32)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        sky += hexcol('#030814', 1.0) + np.exp(-(self.horizon - yy).clip(0) / 0.25) * hexcol('#0a2a2a', 0.4)
        _aurora(sky, t, self.seed, self.horizon, self.get('layers', 3), self.get('speed', 1.0), self.get('hue', 0.0))
        img = resize(sky, W, H) * self.get('gain', 1.0)
        cam = look_at((0, 0, 0), (0, 0.45, 1), (0, 1, 0))
        self.stars.draw(img, cam, 75, t, gain=0.8)
        hz = int(self.horizon * H)
        # reflejo en un lago con ondulación
        refl_h = H - hz
        src = img[max(hz - refl_h, 0):hz][::-1]
        if src.shape[0] < refl_h:
            src = np.pad(src, ((0, refl_h - src.shape[0]), (0, 0), (0, 0)))
        yi = np.arange(refl_h)[:, None]
        xi = np.arange(W)[None, :]
        wob = (np.sin(yi * 0.5 + t * 2.0 + np.sin(xi * 0.013) * 2) * (1 + yi * 0.012)).astype(np.int64)
        xs2 = np.clip(xi + wob, 0, W - 1)
        refl = src[np.broadcast_to(yi, (refl_h, W)), xs2] * 0.5
        img[hz:] = refl * np.linspace(0.9, 0.35, refl_h, dtype=np.float32)[:, None, None]
        # montañas
        yi = np.arange(H, dtype=np.float32)[:, None]
        mt = (yi > self.ridge[None, :]) & (yi <= hz + 2)
        img[mt] = hexcol('#010204', 0.4)
        # reflejo de montañas
        rr = 2 * hz - self.ridge
        mr = (yi > hz) & (yi < rr[None, :])
        img[mr] *= 0.12
        return img


# ───────────────────────────── La Tierra desde órbita ─────────────────────────────

@njit(parallel=True, cache=True)
def _earth(out, t, cam_h, pitch, rot, seed, aur_gain, city_gain):
    h, w, _ = out.shape
    asp = w / h
    R = 1.0
    ce = 1.0 + cam_h
    atm = 1.02
    fov = 0.95
    cp = math.cos(pitch); sp = math.sin(pitch)
    horizon_a = math.acos(1.0 / ce)
    for j in prange(h):
        for i in range(w):
            sx = (i / w - 0.5) * asp * fov
            sy = -(j / h - 0.5) * fov
            dx = sx; dy = sy; dz = 1.0
            dy2 = dy * cp - dz * sp; dz2 = dy * sp + dz * cp
            n = math.sqrt(dx * dx + dy2 * dy2 + dz2 * dz2)
            dx /= n; dy2 /= n; dz2 /= n
            oy = ce
            r = 0.0; g = 0.0; b = 0.0
            bq = oy * dy2
            cq = oy * oy - atm * atm
            disc = bq * bq - cq
            if disc > 0:
                sq = math.sqrt(disc)
                t0 = -bq - sq; t1 = -bq + sq
                dg = bq * bq - (oy * oy - R * R)
                tg = 1e9
                hit = False
                if dg > 0:
                    tg = -bq - math.sqrt(dg)
                    hit = tg > 0
                tend = min(t1, tg)
                tstart = max(t0, 0.0)
                steps = 40
                dt = (tend - tstart) / steps
                for s in range(steps):
                    tt = tstart + (s + 0.5) * dt
                    px = dx * tt; py = oy + dy2 * tt; pz = dz2 * tt
                    rr = math.sqrt(px * px + py * py + pz * pz)
                    alt = (rr - R) / (atm - R)
                    al = math.acos(min(max(py / rr, -1.0), 1.0))       # ángulo desde el nadir
                    az = math.atan2(px, pz) + rot
                    # cortina auroral a una distancia angular ondulada
                    wav = fbm3(az * 3.0, t * 0.05, 0.0, 4, seed)
                    ac = horizon_a * (0.8 + 0.3 * (wav - 0.5) * 2.0)
                    dcur = abs(al - ac)
                    cur = math.exp(-dcur * dcur / 2.5e-5)
                    rays = 0.2 + 1.6 * fbm3(az * 140.0 + wav * 6, t * 0.3, 1.0, 2, seed + 7) ** 2
                    hgt = math.exp(-max(alt - 0.18, 0.0) * 3.0) * (alt > 0.12)
                    em = cur * rays * hgt * aur_gain * 2.0
                    low = math.exp(-(alt - 0.18) * 4.0)
                    r += em * (0.05 * low + 0.9 * max(alt - 0.4, 0.0)) * dt * 60
                    g += em * (1.0 * low) * dt * 60
                    b += em * (0.25 * low + 0.4 * max(alt - 0.4, 0.0)) * dt * 60
                    # capa de airglow (~95 km)
                    ag = math.exp(-((alt - 0.42) ** 2) / 0.0015)
                    r += ag * 0.02 * dt * 60; g += ag * 0.05 * dt * 60; b += ag * 0.01 * dt * 60
                    # dispersión azul fina sobre el limbo
                    sc = math.exp(-alt * 5.0)
                    r += sc * 0.002 * dt * 60; g += sc * 0.005 * dt * 60; b += sc * 0.02 * dt * 60
                if hit:
                    px = dx * tg; py = oy + dy2 * tg; pz = dz2 * tg
                    ux = px; uy = py; uz = pz
                    # rotar la superficie lentamente
                    cr = math.cos(rot * 0.5); sr = math.sin(rot * 0.5)
                    qx = ux * cr - uz * sr; qz = ux * sr + uz * cr
                    land = fbm3(qx * 2.2 + 3, uy * 2.2, qz * 2.2, 6, seed + 30)
                    cloud = fbm3(qx * 6.0, uy * 6.0, qz * 6.0 + t * 0.01, 5, seed + 70)
                    cl = 0.0
                    if land > 0.5:
                        pop = fbm3(qx * 9, uy * 9, qz * 9, 3, seed + 55)
                        city = vnoise3(qx * 1400, uy * 1400, qz * 1400, seed + 50) * vnoise3(qx * 300, uy * 300, qz * 300, seed + 51)
                        cl = max(city - 0.42, 0.0) * 6.0 * max(pop - 0.5, 0.0) * 4.0 * min((land - 0.5) * 30, 1.0) * city_gain
                        r += 0.0015; g += 0.0014; b += 0.0016
                    else:
                        r += 0.0004; g += 0.0006; b += 0.0016
                    cov = max(cloud - 0.5, 0.0) * 2.0
                    r += cl * (1 - 0.7 * cov) * 1.0 + cov * 0.003
                    g += cl * (1 - 0.7 * cov) * 0.55 + cov * 0.004
                    b += cl * (1 - 0.7 * cov) * 0.2 + cov * 0.008
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b


class EarthOrbit(Scene):
    scale = 0.5

    def prepare(self, dur):
        self.stars = StarLayer(12000, self.rng, 0.7)

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        _earth(im, t, self.get('alt', 0.06), self.get('pitch', 0.2) + self.get('tilt', 0.03) * u,
               self.get('rot', 0.0) + t * 0.012, self.seed, self.get('aurora', 1.0), self.get('city', 1.0))
        img = resize(im, W, H)
        st = canvas()
        cam = look_at((0, 0, 0), (0.1 * u, 0.3, 1), (0, 1, 0))
        self.stars.draw(st, cam, 60, t, gain=0.7)
        # estrellas sólo por encima del limbo
        lum = blur(img.sum(2), 2)[..., None]
        img += st * np.clip(1 - lum * 200, 0, 1)
        return img
