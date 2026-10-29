"""
Escenas de física: medusas espaciales, cascadas de rayos cósmicos, muones,
neutrinos y luz de Cherenkov, púlsares, agujero negro, telescopios (JWST,
campo profundo del Hubble), fondo cósmico de microondas, relatividad
(diagrama de Minkowski, curvatura del espacio-tiempo) y muerte térmica.
"""
import math
import numpy as np
from numba import njit, prange

from .core import (W, H, canvas, resize, blur, splat, splat_fast, blackbody, fbm3, ridged3, vnoise3, _hash,
                   add_glow_point, hexcol, ease_in_out, ease_out, ease_in, look_at, project, rot_x, rot_y, rot_z,
                   line, polyline, polyline_fade, disc, smoothstep)
from .scene import Scene
from .scenes_sky import StarLayer, _ridge_line


def _c(h, g=1.0):
    return hexcol(h, g)


# ───────────────────────────── Medusa espacial ─────────────────────────────

@njit(parallel=True, cache=True)
def _jelly(out, t, grow, cx, cy, seed, horizon):
    h, w, _ = out.shape
    asp = w / h
    for j in prange(h):
        for i in range(w):
            x = (i / w - 0.5) * asp
            y = j / h
            # cielo crepuscular
            k = min(max((y - 0.05) / (horizon - 0.05), 0.0), 1.0)
            r = 0.003 + 0.20 * k ** 3.5 + 0.03 * k
            g = 0.006 + 0.08 * k ** 3.0 + 0.03 * k
            b = 0.03 + 0.10 * k ** 1.5
            # campana: paraboloide iluminado por el Sol (por debajo del horizonte)
            dx = (x - cx) / (0.06 + 0.32 * grow)
            dy = (y - cy) / (0.05 + 0.26 * grow)
            rr = math.sqrt(dx * dx + dy * dy)
            ang = math.atan2(dy, dx)
            wob = 0.06 * (fbm3(ang * 2.0, t * 0.2, 0.0, 3, seed) - 0.5)
            openb = 1.0 / (1.0 + math.exp((dy - 0.15) * 14.0))
            shell = math.exp(-((rr - 1.0 - wob) ** 2) / (0.004 + 0.05 * max(dy + 0.6, 0.0) ** 2)) * openb
            inner = math.exp(-rr * rr * 1.5) * 0.3 * openb
            tex = 0.5 + fbm3(dx * 3.0, dy * 3.0, t * 0.1, 5, seed + 3)
            bell = (shell * 0.9 + inner) * tex * (0.2 + 0.8 * min(grow * 3, 1.0))
            # tentáculos: estelas descendentes con estrías
            if dy > 0.0:
                ux = dx / (1.0 + 0.4 * dy)
                strands = fbm3(ux * 14.0, dy * 0.6 - t * 0.05, 1.0, 3, seed + 9)
                strands = max(strands - 0.42, 0.0) * 3.0
                env = math.exp(-ux * ux * 1.8) * math.exp(-dy * 0.9) * (dy < 3.0)
                ten = strands * env * 0.5 * min(grow * 2, 1.0) * min(dy / 0.35, 1.0) ** 2
            else:
                ten = 0.0
            # gradiente de color: núcleo blanco azulado, bordes cian/magenta
            edge = min(rr, 1.4) / 1.4
            cr = 0.7 + 0.5 * edge * edge; cg = 0.85 - 0.25 * edge; cb = 1.0
            r += bell * cr * 0.9 + ten * 0.35
            g += bell * cg * 0.9 + ten * 0.6
            b += bell * cb * 1.0 + ten * 1.0
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b


class SpaceJellyfish(Scene):
    """El fenómeno de la 'medusa espacial': la estela de un cohete iluminada por el Sol al crepúsculo."""
    scale = 0.5

    def prepare(self, dur):
        self.stars = StarLayer(6000, self.rng, 0.5)
        xs = np.arange(W, dtype=np.float64) / W
        self.horizon = self.get('horizon', 0.86)
        self.ridge = _ridge_line(xs, self.seed + 7, 2.0, 0.05 * H, self.horizon * H)

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        grow = ease_out(u * self.get('grow_speed', 1.0), 2)
        cy = 0.42 - 0.08 * grow
        _jelly(im, t, grow, 0.03, cy, self.seed, self.horizon)
        img = resize(im, W, H)
        cam = look_at((0, 0, 0), (0, 0.5, 1), (0, 1, 0))
        st = canvas()
        self.stars.draw(st, cam, 70, t, gain=0.5, bright_glow=False)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img += st * np.clip(1 - yy * 1.4, 0, 1)
        # cohete: trayectoria curva y punto brillante en la punta
        tt = np.linspace(0, 1, 120)
        k = min(u * 1.1 + 0.25, 1.0)
        px = (0.5 + 0.04 * tt ** 2) * W
        py = (self.horizon - (self.horizon - (cy * 1.0)) * tt) * H
        n = max(int(120 * k), 2)
        pts = np.stack([px[:n], py[:n]], 1).astype(np.float32)
        al = (np.linspace(0.2, 1.0, n - 1) ** 1.5).astype(np.float32)
        polyline_fade(img, pts, al, 1.0, 0.75, 0.5, 2.0, 5.0)
        add_glow_point(img, pts[-1, 0], pts[-1, 1], _c('#fff1d6', 2.0), 4.0)
        yi = np.arange(H, dtype=np.float32)[:, None]
        img[yi > self.ridge[None, :]] = _c('#010103', 0.3)
        return img


# ───────────────────────────── Cascada de rayos cósmicos ─────────────────────────────

class CosmicShower(Scene):
    """Un protón de altísima energía choca con la atmósfera y desencadena una lluvia de partículas."""

    def prepare(self, dur):
        rng = self.rng
        segs = []   # x0,y0,x1,y1,t0,t1,tipo
        top = 0.30 * H

        def branch(x, y, ang, t0, energy, kind, depth):
            L = (60 + 140 * rng.random()) * (0.6 + energy * 0.5)
            speed = 900.0
            x1 = x + math.sin(ang) * L; y1 = y + math.cos(ang) * L
            t1 = t0 + L / speed
            segs.append((x, y, x1, y1, t0, t1, kind))
            if y1 > H * 1.05 or depth > 9:
                return
            if kind == 1:   # muón: atraviesa sin ramificar hasta el suelo
                branch(x1, y1, ang + rng.normal(0, 0.02), t1, energy, 1, depth + 1)
                return
            nchild = 2 if energy > 0.15 else (1 if rng.random() < 0.7 else 0)
            for c in range(nchild + (1 if depth < 3 else 0)):
                e2 = energy * rng.uniform(0.35, 0.65)
                k2 = 1 if (rng.random() < 0.18 and depth > 1) else (2 if rng.random() < 0.4 else 0)
                branch(x1, y1, ang + rng.normal(0, 0.16 + 0.03 * depth), t1, e2, k2, depth + 1)

        x0 = W * 0.52
        segs.append((x0 - 0.05 * W, -40.0, x0, top, -0.5, 0.0, 3))
        for k in range(3):
            branch(x0, top, rng.normal(0, 0.12), 0.0, 1.0, 0, 0)
        self.segs = np.array(segs, np.float64)
        self.t_hit = self.get('t_hit', 1.0)
        self.colors = {0: _c('#7fb8ff', 1.4), 1: _c('#ffb35c', 1.8), 2: _c('#f2f6ff', 1.0), 3: _c('#ffffff', 3.0)}
        self.stars = StarLayer(5000, rng, 0.5)

    def draw(self, t, u):
        img = canvas()
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        # atmósfera: degradado azul profundo, más denso abajo
        img += (np.clip(yy - 0.25, 0, 1) ** 1.6) * _c('#1d4f9a', 0.35) + _c('#020510', 1.0)
        cam = look_at((0, 0, 0), (0, 0.6, 1), (0, 1, 0))
        st = canvas(); self.stars.draw(st, cam, 70, t, gain=0.5, bright_glow=False)
        img += st * np.clip(1 - yy * 2.5, 0, 1)
        # capas atmosféricas sutiles
        for k, yl in enumerate((0.3, 0.48, 0.66, 0.84)):
            img[int(yl * H)] += _c('#4c7bd1', 0.05)
        tl = t - self.t_hit
        for s in self.segs:
            x0, y0, x1, y1, t0, t1, kind = s
            if tl < t0:
                continue
            f = min((tl - t0) / max(t1 - t0, 1e-6), 1.0)
            xe = x0 + (x1 - x0) * f; ye = y0 + (y1 - y0) * f
            age = tl - t1
            fade = 1.0 if age < 0 else math.exp(-age * 0.7)
            c = self.colors[int(kind)] * fade
            wdt = 2.2 if kind == 3 else (1.6 if kind == 1 else 1.1)
            line(img, x0, y0, xe, ye, c[0], c[1], c[2], wdt, 3.0)
            if f < 1.0:
                add_glow_point(img, xe, ye, c * 0.9, 2.5)
        # destello del primer impacto
        if tl > -0.05:
            fl = math.exp(-max(tl, 0) * 2.5)
            add_glow_point(img, W * 0.52, 0.30 * H, _c('#e6f0ff', 6.0 * fl), 18.0, spikes=4, spike_len=300 * fl)
        return img


# ───────────────────────────── Cherenkov / Super-Kamiokande ─────────────────────────────

@njit(parallel=True, cache=True)
def _pmt_wall(out, t, yaw, pitch, ring_t, ring_dir, ring_cx, ring_cy, cols, rows, seed):
    h, w, _ = out.shape
    asp = w / h
    cy_ = math.cos(yaw); sy_ = math.sin(yaw)
    cp = math.cos(pitch); sp = math.sin(pitch)
    for j in prange(h):
        for i in range(w):
            dx = (i / w - 0.5) * asp * 1.2
            dy = -(j / h - 0.5) * 1.2
            dz = 1.0
            # pitch
            dy2 = dy * cp + dz * sp; dz2 = -dy * sp + dz * cp
            # yaw
            dx2 = dx * cy_ + dz2 * sy_; dz3 = -dx * sy_ + dz2 * cy_
            n = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz3 * dz3)
            dx2 /= n; dy2 /= n; dz3 /= n
            # cilindro de radio 1 alrededor del eje y, cámara en el centro (0,-0.3,0)
            a = dx2 * dx2 + dz3 * dz3
            tc = 1.0 / math.sqrt(max(a, 1e-9))
            py = -0.3 + dy2 * tc
            r = 0.0; g = 0.0; b = 0.0
            if abs(py) < 1.6:
                px = dx2 * tc; pz = dz3 * tc
                th = math.atan2(px, pz)
                u = th / (2 * math.pi) * cols
                v = py / 3.2 * rows
                fu = u - math.floor(u) - 0.5; fv = v - math.floor(v) - 0.5
                d = math.sqrt(fu * fu + fv * fv)
                cell_u = math.floor(u); cell_v = math.floor(v)
                # fotomultiplicador: cúpula dorada con reflejo
                if d < 0.42:
                    dome = math.sqrt(max(1.0 - (d / 0.42) ** 2, 0.0))
                    spec = math.exp(-((fu + 0.14) ** 2 + (fv - 0.14) ** 2) / 0.004) * 0.6
                    base = 0.02 + 0.05 * dome
                    r = base * 1.0 + spec * 0.9; g = base * 0.75 + spec * 0.8; b = base * 0.35 + spec * 0.6
                    # anillo de Cherenkov
                    cu = (cell_u + 0.5) / cols * 2 * math.pi
                    cv = (cell_v + 0.5) / rows * 3.2
                    du = math.atan2(math.sin(cu - ring_cx), math.cos(cu - ring_cx))
                    dv = cv - ring_cy
                    rd = math.sqrt(du * du + dv * dv * 0.9)
                    ring_r = 0.28 * min(ring_t, 1.0)
                    hitv = math.exp(-((rd - ring_r) ** 2) / 0.0012) * (ring_t > 0)
                    flick = 0.6 + 0.8 * _hash(int(cell_u), int(cell_v), int(t * 12), seed)
                    fill = math.exp(-((rd - ring_r * 0.85) ** 2) / 0.006) * 0.2 * (ring_t > 0) * flick
                    e = (hitv * flick + fill) * math.exp(-max(ring_t - 1.0, 0.0) * 1.2)
                    r += e * 0.45 * dome; g += e * 0.75 * dome; b += e * 1.6 * dome
                else:
                    r = 0.004; g = 0.0045; b = 0.006
            else:
                r = 0.002; g = 0.002; b = 0.003
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b


class Cherenkov(Scene):
    """Interior de un detector de neutrinos: una pared de fotomultiplicadores dorados y un anillo de luz azul."""
    scale = 0.5

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        ring_t = (t - self.get('t_event', 1.2)) / 0.6
        _pmt_wall(im, t, self.get('yaw', 0.0) + 0.06 * u, self.get('pitch', 0.12), ring_t, 0.0,
                  self.get('ring_x', 0.05), self.get('ring_y', 0.15), 110, 40, self.seed)
        img = resize(im, W, H)
        # destello azul del cono Cherenkov
        if ring_t > 0:
            fl = math.exp(-ring_t * 1.5)
            img += blur(img, 25) * 1.5 * fl
        return img


# ───────────────────────────── Neutrinos atravesando la Tierra ─────────────────────────────

class NeutrinoEarth(Scene):
    def prepare(self, dur):
        rng = self.rng
        n = self.get('n', 900)
        self.y0 = rng.uniform(-0.1, 1.1, n) * H
        self.speed = rng.uniform(1.6, 2.4, n) * W
        self.off = rng.random(n)
        self.len = rng.uniform(80, 260, n)
        self.bright = rng.uniform(0.15, 0.5, n)
        self.stars = StarLayer(9000, rng, 0.6)
        # puntos de superficie de la Tierra (malla de latitudes / longitudes)
        self.lat_lines = [np.radians(a) for a in range(-75, 90, 15)]
        self.lon_lines = [np.radians(a) for a in range(0, 360, 15)]

    def draw(self, t, u):
        img = canvas()
        cam = look_at((0, 0, 0), (0, 0, 1), (0, 1, 0))
        self.stars.draw(img, cam, 60, t, gain=0.6)
        cx, cy, R = W * 0.5, H * 0.52, W * 0.36
        # Sol a la izquierda
        add_glow_point(img, -W * 0.25, cy, _c('#ffd9a0', 3.0), 260)
        # Tierra: globo translúcido de alambre
        rot = t * 0.15
        tilt = 0.4
        for lat in self.lat_lines:
            th = np.linspace(0, 2 * np.pi, 180)
            P = np.stack([np.cos(lat) * np.cos(th + rot), np.sin(lat) * np.ones_like(th), np.cos(lat) * np.sin(th + rot)], 1)
            self._draw_curve(img, P, cx, cy, R, tilt)
        for lon in self.lon_lines:
            ph = np.linspace(-np.pi / 2, np.pi / 2, 90)
            P = np.stack([np.cos(ph) * np.cos(lon + rot), np.sin(ph), np.cos(ph) * np.sin(lon + rot)], 1)
            self._draw_curve(img, P, cx, cy, R, tilt)
        yy, xx = np.mgrid[0:H, 0:W]
        dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        rim = np.exp(-((dd - 1.0) ** 2) / 0.0012) * 0.6 + np.clip(1 - dd, 0, 1) ** 0.5 * 0.03
        img += rim[..., None].astype(np.float32) * _c('#4aa8ff', 1.0)
        # neutrinos: trazos rectos que lo atraviesan todo sin desviarse
        x = ((self.off + t * self.speed / W) % 1.3) * W * 1.4 - W * 0.3
        for i in range(self.y0.size):
            xe = x[i]; xs = xe - self.len[i]
            b = self.bright[i]
            line(img, xs, self.y0[i], xe, self.y0[i], 0.55 * b, 0.75 * b, 1.0 * b, 0.9, 0.0)
        return img

    @staticmethod
    def _draw_curve(img, P, cx, cy, R, tilt):
        P = P @ rot_x(tilt).T
        front = P[:, 2] < 0
        X = cx + P[:, 0] * R; Y = cy - P[:, 1] * R
        pts = np.stack([X, Y], 1).astype(np.float32)
        al = np.where(front[:-1], 0.5, 0.12).astype(np.float32)
        polyline_fade(img, pts, al, 0.25, 0.55, 1.0, 1.0, 2.0)


# ───────────────────────────── Púlsar ─────────────────────────────

class Pulsar(Scene):
    def prepare(self, dur):
        self.stars = StarLayer(14000, self.rng, 0.7)
        self.period = self.get('period', 1.0)
        self.incl = self.get('incl', 0.55)
        # líneas de campo dipolares r = L sin²θ en varios meridianos
        self.lines = []
        for L in (1.8, 2.6, 3.8, 5.5):
            for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                th = np.linspace(0.02, np.pi - 0.02, 160)
                r = L * np.sin(th) ** 2
                P = np.stack([r * np.sin(th) * np.cos(phi), r * np.cos(th), r * np.sin(th) * np.sin(phi)], 1)
                self.lines.append((P, L))

    def draw(self, t, u):
        img = canvas()
        orbit = 0.15 * t
        cam = look_at((math.sin(orbit) * 14, 2.0, -math.cos(orbit) * 14), (0, 0, 0), (0, 1, 0))
        fov = 50
        self.stars.draw(img, cam, fov, t, gain=0.7)
        spin = 2 * math.pi * t / self.period
        Rm = rot_y(spin) @ rot_z(self.incl)
        # haces de radiación (conos) a lo largo del eje magnético
        axis = Rm @ np.array([0, 1.0, 0])
        for sgn in (1, -1):
            a = axis * sgn
            tip = a * 16
            x0, y0, _ = project(np.zeros((1, 3)), cam, fov)
            ts = np.linspace(0, 1, 40)
            for k in range(-4, 5):
                perp = np.cross(a, cam[3]); perp /= max(np.linalg.norm(perp), 1e-6)
                spread = k * 0.06
                P = np.outer(ts, tip) + np.outer(ts * 16 * spread, perp)
                x, y, z = project(P, cam, fov)
                pts = np.stack([x, y], 1).astype(np.float32)
                facing = max(np.dot(a, -cam[3]), 0)
                inten = (0.06 + 0.5 * facing ** 6) * math.exp(-abs(k) * 0.35)
                al = ((1 - ts[:-1]) ** 1.5 * inten).astype(np.float32)
                polyline_fade(img, pts, al, 0.55, 0.75, 1.0, 3.0, 9.0)
        # líneas de campo
        for P, L in self.lines:
            Q = P @ Rm.T
            x, y, z = project(Q, cam, fov)
            pts = np.stack([x, y], 1).astype(np.float32)
            al = np.full(len(pts) - 1, 0.18 / math.sqrt(L), np.float32)
            polyline_fade(img, pts, al, 0.45, 0.6, 1.0, 1.0, 2.0)
        # estrella de neutrones
        x, y, _ = project(np.zeros((1, 3)), cam, fov)
        facing = max(np.dot(axis, -cam[3]), np.dot(-axis, -cam[3]), 0)
        flash = facing ** 12
        add_glow_point(img, x[0], y[0], _c('#cfe2ff', 3.0 + 18 * flash), 9.0, spikes=4, spike_len=80 + 600 * flash)
        if flash > 0.05:
            img += _c('#a9c8ff', 0.15 * flash)
        return img


# ───────────────────────────── Agujero negro (Schwarzschild) ─────────────────────────────

@njit(parallel=True, cache=True)
def _blackhole(out, t, cam_r, cam_el, fov, seed, disk_in, disk_out, steps):
    h, w, _ = out.shape
    asp = w / h
    ce = math.cos(cam_el); se = math.sin(cam_el)
    for j in prange(h):
        for i in range(w):
            sx = (i / w - 0.5) * asp * fov
            sy = -(j / h - 0.5) * fov
            # cámara en (0, r sin el, -r cos el) mirando al origen
            cx = 0.0; cy = cam_r * se; cz = -cam_r * ce
            fx = 0.0; fy = -se; fz = ce
            ux = 0.0; uy = ce; uz = se
            rx = 1.0; ry = 0.0; rz = 0.0
            dx = fx + sx * rx + sy * ux; dy = fy + sx * ry + sy * uy; dz = fz + sx * rz + sy * uz
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n; dy /= n; dz /= n
            px = cx; py = cy; pz = cz
            # momento angular h² = |p × v|²
            hx = py * dz - pz * dy; hy = pz * dx - px * dz; hz = px * dy - py * dx
            h2 = hx * hx + hy * hy + hz * hz
            r = 0.0; g = 0.0; b = 0.0
            alpha = 1.0
            captured = False
            for s in range(steps):
                rr2 = px * px + py * py + pz * pz
                rr = math.sqrt(rr2)
                dt = 0.04 * rr if rr < 12 else 0.6
                if rr < 1.0:
                    captured = True
                    break
                acc = -1.5 * h2 / (rr2 * rr2 * rr)
                nx = px + dx * dt; ny = py + dy * dt; nz = pz + dz * dt
                dx += acc * px * dt; dy += acc * py * dt; dz += acc * pz * dt
                # cruce del plano del disco
                if py * ny < 0:
                    f = py / (py - ny)
                    ix = px + (nx - px) * f; iz = pz + (nz - pz) * f
                    ir = math.sqrt(ix * ix + iz * iz)
                    if ir > disk_in and ir < disk_out:
                        ang = math.atan2(iz, ix)
                        # rotación kepleriana + turbulencia
                        om = 1.0 / (ir ** 1.5)
                        tex = fbm3(ir * 3.0, (ang + t * om * 2.0) * 3.0, 0.0, 4, seed)
                        tex2 = fbm3(ir * 12.0, (ang + t * om * 2.0) * 10.0, 1.0, 3, seed + 3)
                        temp = (disk_in / ir) ** 0.75
                        # Doppler relativista aproximado (lado que se acerca más brillante)
                        vx = -math.sin(ang); vz = math.cos(ang)
                        beta = 0.55 / math.sqrt(ir)
                        dop = 1.0 / (1.0 - beta * (vx * dx + vz * dz) / math.sqrt(dx * dx + dy * dy + dz * dz))
                        I = temp ** 2.5 * (0.35 + 1.3 * tex * tex2) * dop ** 3.0
                        edge = min((ir - disk_in) / 0.6, 1.0) * min((disk_out - ir) / 3.0, 1.0)
                        I *= edge * 2.0
                        tc = min(temp * dop, 1.6)
                        cr = 1.0; cg = 0.45 + 0.4 * tc; cb = 0.15 + 0.55 * tc * tc
                        a = min(I, 1.0) * 0.9
                        r += alpha * I * cr; g += alpha * I * cg; b += alpha * I * cb
                        alpha *= (1.0 - a)
                        if alpha < 0.02:
                            break
                px = nx; py = ny; pz = nz
                if rr > cam_r * 1.6:
                    break
            if not captured and alpha > 0.02:
                # fondo: estrellas en la dirección final
                n = math.sqrt(dx * dx + dy * dy + dz * dz)
                qx = dx / n; qy = dy / n; qz = dz / n
                sc = 180.0
                cxs = math.floor(qx * sc); cys = math.floor(qy * sc); czs = math.floor(qz * sc)
                hv = _hash(int(cxs), int(cys), int(czs), seed + 11)
                if hv > 0.985:
                    s = (hv - 0.985) * 66.0
                    r += alpha * s * 1.4; g += alpha * s * 1.4; b += alpha * s * 1.6
                neb = fbm3(qx * 3.0, qy * 3.0, qz * 3.0, 4, seed + 20)
                r += alpha * max(neb - 0.5, 0.0) * 0.08; g += alpha * max(neb - 0.5, 0.0) * 0.04; b += alpha * max(neb - 0.5, 0.0) * 0.12
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b


class BlackHole(Scene):
    scale = 0.42

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        r0, r1 = self.get('dist', (26.0, 19.0))
        el = self.get('elev', (0.12, 0.2))
        _blackhole(im, t, r0 + (r1 - r0) * ease_in_out(u), el[0] + (el[1] - el[0]) * u, self.get('fov', 0.75),
                   self.seed, 3.0, 14.0, 360)
        return resize(im, W, H) * self.get('gain', 1.0)


# ───────────────────────────── JWST: espejo de berilio dorado ─────────────────────────────

@njit(parallel=True, cache=True)
def _jwst(out, t, rot, tilt, unfold, scale, seed, shine):
    h, w, _ = out.shape
    asp = w / h
    sq3 = math.sqrt(3.0)
    ct = math.cos(tilt); st = math.sin(tilt)
    for j in prange(h):
        for i in range(w):
            x = (i / w - 0.5) * asp / scale
            y = (j / h - 0.5) / scale
            # inclinación (proyección perspectiva simple sobre el eje x)
            y2 = y / (ct - y * st * 0.5)
            x2 = x / (1 - y * st * 0.5)
            cr = math.cos(rot); sr = math.sin(rot)
            xr = x2 * cr - y2 * sr; yr = x2 * sr + y2 * cr
            # coordenadas hexagonales axiales (hexágonos con vértice arriba)
            q = (sq3 / 3 * xr - yr / 3) / 0.5
            r_ = (2.0 / 3.0 * yr) / 0.5
            # redondeo cúbico
            cx = q; cz = r_; cy = -cx - cz
            rx = round(cx); ry = round(cy); rz = round(cz)
            dx = abs(rx - cx); dy = abs(ry - cy); dz = abs(rz - cz)
            if dx > dy and dx > dz:
                rx = -ry - rz
            elif dy > dz:
                ry = -rx - rz
            else:
                rz = -rx - ry
            ring = max(abs(rx), abs(ry), abs(rz))
            R = 0.0; G = 0.0; B = 0.0
            if ring >= 1 and ring <= 2:
                # alas laterales plegables
                wing = (rx >= 2 and rz <= -1) or (rx <= -2 and rz >= 1)
                vis = True
                if wing and unfold < 1.0:
                    vis = unfold > 0.02
                # centro del hexágono
                hx = 0.5 * (sq3 * rx + sq3 / 2 * rz)
                hy = 0.5 * (1.5 * rz)
                lx = xr - hx; ly = yr - hy
                # distancia al borde del hexágono (con hueco)
                ax = abs(lx); ay = abs(ly)
                dh = max(ax, ax * 0.5 + ay * 0.866) / (0.5 * 0.866)
                if vis and dh < 0.955:
                    # normal ligeramente curva -> reflejos que se desplazan
                    nx = lx * 0.35 + 0.02 * (_hash(int(rx), int(rz), 0, seed) - 0.5)
                    ny = ly * 0.35 + 0.02 * (_hash(int(rx), int(rz), 1, seed) - 0.5)
                    env = fbm3(nx * 6 + xr * 0.6 + t * 0.05, ny * 6 + yr * 0.6, 0.0, 4, seed + 3)
                    sweep = math.exp(-((xr * 0.7 + yr * 0.7 - shine) ** 2) / 0.05)
                    bevel = max(dh - 0.88, 0.0) / 0.075
                    base = 0.18 + 0.6 * env * env + 1.6 * sweep + 0.8 * bevel * bevel
                    seg_var = 0.85 + 0.3 * _hash(int(rx), int(rz), 2, seed)
                    if wing and unfold < 1.0:
                        base *= unfold
                    R = base * 1.0 * seg_var; G = base * 0.66 * seg_var; B = base * 0.22 * seg_var
            # puntales del espejo secundario
            for k in range(3):
                a = -math.pi / 2 + k * 2 * math.pi / 3 + (0.0 if k == 0 else 0.0)
                ux = math.cos(a); uy = math.sin(a)
                along = xr * ux + yr * uy
                perp = abs(-xr * uy + yr * ux)
                if along > 0.05 and along < 2.0 and perp < 0.012:
                    R = 0.01; G = 0.01; B = 0.012
            out[j, i, 0] = R; out[j, i, 1] = G; out[j, i, 2] = B


class JWSTMirror(Scene):
    scale = 0.5

    def prepare(self, dur):
        self.stars = StarLayer(9000, self.rng, 0.6)

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        unf = self.get('unfold', None)
        unfold = 1.0 if unf is None else ease_in_out((u - unf[0]) / (unf[1] - unf[0]))
        s0, s1 = self.get('zoom', (0.17, 0.21))
        _jwst(im, t, self.get('rot', 0.0) + 0.04 * u, self.get('tilt', 0.35) - 0.12 * u, unfold,
              s0 + (s1 - s0) * ease_in_out(u), self.seed, -1.6 + 3.2 * u)
        img = resize(im, W, H)
        bg = canvas()
        cam = look_at((0, 0, 0), (0, 0, 1), (0, 1, 0))
        self.stars.draw(bg, cam, 60, t, gain=0.6)
        m = (img.sum(2, keepdims=True) < 0.01)
        img += bg * m
        return img


# ───────────────────────────── Campo profundo (Hubble / Webb) ─────────────────────────────

@njit(cache=True)
def _galaxies(img, xs, ys, sizes, angs, ell, cols, arms, lums):
    h, w, _ = img.shape
    for k in range(xs.shape[0]):
        s = sizes[k]
        R = int(s * 5 + 2)
        cx = xs[k]; cy = ys[k]
        if cx < -R or cx > w + R or cy < -R or cy > h + R:
            continue
        ca = math.cos(angs[k]); sa = math.sin(angs[k])
        for yy in range(max(int(cy - R), 0), min(int(cy + R), h)):
            for xx in range(max(int(cx - R), 0), min(int(cx + R), w)):
                dx = xx + 0.5 - cx; dy = yy + 0.5 - cy
                u = (dx * ca + dy * sa) / s
                v = (-dx * sa + dy * ca) / (s * ell[k])
                r = math.sqrt(u * u + v * v) + 1e-6
                I = math.exp(-r * 1.6) * 1.0 + math.exp(-r * r * 12) * 1.5
                if arms[k] > 0:
                    th = math.atan2(v, u)
                    sp = 0.5 + 0.5 * math.cos(2 * th - 4.0 * math.log(r + 0.05))
                    I *= 0.45 + 0.9 * sp * min(r * 2.5, 1.0)
                I *= lums[k]
                if I < 1e-4:
                    continue
                img[yy, xx, 0] += cols[k, 0] * I
                img[yy, xx, 1] += cols[k, 1] * I
                img[yy, xx, 2] += cols[k, 2] * I


class DeepField(Scene):
    def prepare(self, dur):
        rng = self.rng
        n = self.get('n', 3200)
        self.gx = rng.uniform(-0.1, 1.1, n); self.gy = rng.uniform(-0.1, 1.1, n)
        self.depth = rng.uniform(0.3, 1.0, n)
        self.gs = (rng.pareto(2.0, n) + 1) * 4.0 * self.depth
        self.gs = np.minimum(self.gs, 70)
        self.ga = rng.random(n) * np.pi
        self.ge = rng.uniform(0.25, 1.0, n)
        z = rng.random(n)
        palette = np.array([_c('#ffd7a3'), _c('#ff9a5c'), _c('#ff6b4a'), _c('#a8c8ff'), _c('#fff2dd'), _c('#d07bff')])
        pick = np.where(z < 0.35, 2, np.where(z < 0.55, 1, np.where(z < 0.7, 0, np.where(z < 0.85, 3, np.where(z < 0.96, 4, 5)))))
        self.gc = palette[pick]
        self.garms = (rng.random(n) < 0.35).astype(np.float64)
        self.gl = rng.uniform(0.25, 0.9, n) * (1 + 0.5 * (self.gs > 14))
        self.stars_x = rng.random(12); self.stars_y = rng.random(12)

    def draw(self, t, u):
        img = canvas()
        z0, z1 = self.get('zoom', (1.0, 1.6))
        zoom = z0 * (z1 / z0) ** ease_in_out(u)
        zz = zoom ** (0.5 + self.depth)
        fx, fy = self.get('focus', (0.5, 0.5))
        x = ((self.gx - fx) * zz + 0.5) * W
        y = ((self.gy - fy) * zz + 0.5) * H
        _galaxies(img, x, y, self.gs * zz, self.ga + t * 0.003, self.ge, self.gc, self.garms, self.gl)
        img = img + blur(img, 2.5) * 0.5
        for k in range(6):
            sx = ((self.stars_x[k] - fx) * zoom ** 1.6 + 0.5) * W
            sy = ((self.stars_y[k] - fy) * zoom ** 1.6 + 0.5) * H
            add_glow_point(img, sx, sy, _c('#fff6e8', 1.2), 3.0, spikes=8, spike_len=140, angle=math.pi / 2, spike_w=1.0)
        return img


# ───────────────────────────── Fondo cósmico de microondas ─────────────────────────────

@njit(parallel=True, cache=True)
def _cmb(out, t, mode, lon0, reveal, seed, scale):
    h, w, _ = out.shape
    asp = w / h
    for j in prange(h):
        for i in range(w):
            x = (i / w - 0.5) * asp * 2.0 / scale
            y = (j / h - 0.5) * 2.0 / scale
            inside = False
            sx = 0.0; sy = 0.0; sz = 0.0
            if mode == 0:
                # proyección de Mollweide (elipse 2:1 rotada 90° para el formato vertical)
                X = -y * 1.0; Y = x * 1.0
                if (X / 2.0) ** 2 + Y ** 2 < 1.0:
                    inside = True
                    th = math.asin(min(max(Y, -1.0), 1.0))
                    lat = math.asin(min(max((2 * th + math.sin(2 * th)) / math.pi, -1.0), 1.0))
                    lon = math.pi * X / (2.0 * math.cos(th) + 1e-9) + lon0
                    sx = math.cos(lat) * math.cos(lon); sy = math.sin(lat); sz = math.cos(lat) * math.sin(lon)
                    if X / 2.0 > reveal * 2.0 - 1.0:
                        inside = False
            else:
                rr = x * x + y * y
                if rr < 1.0:
                    inside = True
                    zz = math.sqrt(1.0 - rr)
                    c = math.cos(lon0); s = math.sin(lon0)
                    sx = x * c + zz * s; sz = -x * s + zz * c; sy = -y
            if inside:
                v = 0.0; amp = 1.0; f = 3.0
                for o in range(7):
                    v += amp * (vnoise3(sx * f, sy * f, sz * f, seed + o) - 0.5)
                    f *= 2.0; amp *= 0.72
                v = v * 2.0
                # paleta Planck: azul profundo -> blanco cálido -> rojo
                if v < 0:
                    k = min(-v, 1.0)
                    r = 0.9 - 0.85 * k; g = 0.85 - 0.6 * k; b = 0.75 + 0.2 * k
                else:
                    k = min(v, 1.0)
                    r = 0.95 + 0.05 * k; g = 0.85 - 0.6 * k; b = 0.7 - 0.62 * k
                lim = 1.0
                if mode == 1:
                    lim = 0.55 + 0.45 * math.sqrt(max(1.0 - (x * x + y * y), 0.0))
                out[j, i, 0] = r * r * 0.55 * lim; out[j, i, 1] = g * g * 0.55 * lim; out[j, i, 2] = b * b * 0.55 * lim
            else:
                out[j, i, 0] = 0.0; out[j, i, 1] = 0.0; out[j, i, 2] = 0.0


class CMBMap(Scene):
    scale = 0.5

    def draw(self, t, u):
        w, h = self.size()
        im = np.zeros((h, w, 3), np.float32)
        mode = 0 if self.get('mode', 'mollweide') == 'mollweide' else 1
        rev = ease_in_out(u / self.get('reveal', 0.5)) if mode == 0 else 1.0
        _cmb(im, t, mode, 0.25 * t * self.get('spin', 1.0), rev, self.seed, self.get('size', 0.92 if mode else 0.47))
        img = resize(im, W, H)
        if mode == 1:
            yy, xx = np.mgrid[0:H, 0:W]
            R = 0.92 * 0.5 * H * self.get('size', 0.92) / 1.0
            dd = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * 0.5 * self.get('size', 0.92))
            img += (np.exp(-((dd - 1.0) ** 2) / 0.002) * 0.3)[..., None].astype(np.float32) * _c('#ffb07a')
        return img


# ───────────────────────────── Diagrama de Minkowski (transformación de Lorentz) ─────────────────────────────

class Minkowski(Scene):
    """Ejes (x, ct) en reposo y ejes (x', ct') de un observador con velocidad β: el abanico de Lorentz."""

    def draw(self, t, u):
        img = canvas()
        b0, b1 = self.get('beta', (0.0, 0.8))
        beta = b0 + (b1 - b0) * ease_in_out(min(u * 1.3, 1.0))
        g = 1 / math.sqrt(1 - beta ** 2)
        cx, cy = W * 0.5, H * 0.56
        S = W * 0.11
        reveal = ease_out(min(u * 4, 1.0))

        def P(x, ct):
            return cx + x * S, cy - ct * S

        lim = 5.5
        grid = _c('#3a5a8c', 0.22 * reveal)
        for k in np.arange(-8, 9):
            x0, y0 = P(k, -lim * 1.8); x1, y1 = P(k, lim * 1.8)
            line(img, x0, y0, x1, y1, grid[0], grid[1], grid[2], 1.0, 0.0)
            x0, y0 = P(-lim, k); x1, y1 = P(lim, k)
            line(img, x0, y0, x1, y1, grid[0], grid[1], grid[2], 1.0, 0.0)
        ax = _c('#e8eef8', 0.9 * reveal)
        x0, y0 = P(-lim, 0); x1, y1 = P(lim, 0); line(img, x0, y0, x1, y1, *ax, 2.0, 2.0)
        x0, y0 = P(0, -lim * 1.6); x1, y1 = P(0, lim * 1.6); line(img, x0, y0, x1, y1, *ax, 2.0, 2.0)
        # cono de luz
        lc = _c('#ffc56b', 1.6 * reveal)
        for s in (1, -1):
            x0, y0 = P(-lim * 1.3, -s * lim * 1.3); x1, y1 = P(lim * 1.3, s * lim * 1.3)
            line(img, x0, y0, x1, y1, *lc, 2.4, 7.0)
        # hipérbolas invariantes s² = const
        hy = _c('#9b7bff', 0.35 * reveal)
        for s2 in (1, 4, 9):
            a = math.sqrt(s2)
            tt = np.linspace(-2.2, 2.2, 200)
            for sg in (1, -1):
                pts = np.stack([np.array(P(a * np.sinh(tt), sg * a * np.cosh(tt))[0]),
                                np.array(P(a * np.sinh(tt), sg * a * np.cosh(tt))[1])], 1).astype(np.float32)
                polyline(img, pts, hy[0], hy[1], hy[2], 1.4, 3.0)
        # ejes y rejilla del sistema en movimiento
        bc = _c('#53e0ff', 1.0)
        gb = _c('#53e0ff', 0.18)
        for k in np.arange(-6, 7):
            # líneas de x' = k : x = γ(k + β ct'), ct = γ(ct' + β k)
            tp = np.array([-lim * 2, lim * 2])
            xa, ya = P(g * (k + beta * tp), g * (tp + beta * k))
            line(img, xa[0], ya[0], xa[1], ya[1], gb[0], gb[1], gb[2], 1.0, 0.0)
            xp = np.array([-lim * 2, lim * 2])
            xb, yb = P(g * (xp + beta * k), g * (k + beta * xp))
            line(img, xb[0], yb[0], xb[1], yb[1], gb[0], gb[1], gb[2], 1.0, 0.0)
        xa, ya = P(g * np.array([0, 0]) + g * beta * np.array([-lim, lim]), g * np.array([-lim, lim]))
        line(img, xa[0], ya[0], xa[1], ya[1], *bc, 2.6, 8.0)
        xb, yb = P(g * np.array([-lim, lim]), g * beta * np.array([-lim, lim]))
        line(img, xb[0], yb[0], xb[1], yb[1], *bc, 2.6, 8.0)
        # línea de universo de una partícula y su "tic-tac" propio
        wl = _c('#ff6fa8', 1.4)
        xa, ya = P(beta * np.array([-lim, lim]), np.array([-lim, lim]))
        for k in range(-5, 6):
            tau = k * 1.0
            px, py = P(g * beta * tau, g * tau)
            add_glow_point(img, px, py, wl * 0.5, 3.0)
        add_glow_point(img, *P(0, 0), _c('#ffffff', 1.0), 4.0)
        return img


# ───────────────────────────── Curvatura del espacio-tiempo ─────────────────────────────

class SpacetimeWell(Scene):
    def prepare(self, dur):
        self.stars = StarLayer(8000, self.rng, 0.5)

    def depth(self, X, Z, mass):
        r = np.sqrt(X ** 2 + Z ** 2)
        return -mass / np.sqrt(r ** 2 + 0.35)

    def draw(self, t, u):
        img = canvas()
        ang = 0.25 + 0.12 * t
        el = self.get('elev', 0.5)
        dist = self.get('dist', (9.0, 7.0))
        d = dist[0] + (dist[1] - dist[0]) * ease_in_out(u)
        cam = look_at((math.sin(ang) * d * math.cos(el), d * math.sin(el), -math.cos(ang) * d * math.cos(el)), (0, -0.8, 0))
        fov = 50
        self.stars.draw(img, cam, fov, t, gain=0.5, bright_glow=False)
        m0, m1 = self.get('mass', (0.0, 1.6))
        mass = m0 + (m1 - m0) * ease_in_out(min(u * 1.6, 1.0))
        L = 8.0
        n = 33
        ss = np.linspace(-L, L, 220)
        col = _c('#5ad1ff', 1.0)
        for k in np.linspace(-L, L, n):
            for orient in (0, 1):
                X = ss if orient == 0 else np.full_like(ss, k)
                Z = np.full_like(ss, k) if orient == 0 else ss
                Y = self.depth(X, Z, mass)
                Pp = np.stack([X, Y, Z], 1)
                x, y, z = project(Pp, cam, fov)
                pts = np.stack([x, y], 1).astype(np.float32)
                fade = np.clip(1 - (np.sqrt(X ** 2 + Z ** 2) / L) ** 2, 0, 1)
                al = (0.10 + 0.55 * fade[:-1] * (1 + 0.6 * np.exp(-(X[:-1] ** 2 + Z[:-1] ** 2) / 3))) * 0.6
                polyline_fade(img, pts, al.astype(np.float32), col[0], col[1], col[2], 1.2, 2.5)
        # estrella central
        x, y, _ = project(np.array([[0, self.depth(0, 0, mass) + 0.45, 0.0]]), cam, fov)
        add_glow_point(img, x[0], y[0], _c('#ffd29a', 3.0), 16.0)
        # planeta en órbita con estela
        if self.get('orbit', True):
            ph = t * 0.9
            tr = np.linspace(ph - 2.4, ph, 80)
            R = 3.2
            X = R * np.cos(tr); Z = R * np.sin(tr)
            Y = self.depth(X, Z, mass) + 0.18
            x, y, z = project(np.stack([X, Y, Z], 1), cam, fov)
            pts = np.stack([x, y], 1).astype(np.float32)
            al = np.linspace(0, 1, 79).astype(np.float32) ** 2
            polyline_fade(img, pts, al, 0.6, 0.9, 1.0, 2.0, 5.0)
            add_glow_point(img, x[-1], y[-1], _c('#bfe6ff', 1.5), 5.0)
        return img


# ───────────────────────────── Muerte térmica ─────────────────────────────

class HeatDeath(Scene):
    """Las estrellas se apagan una a una, enrojecen y el universo tiende al equilibrio."""

    def prepare(self, dur):
        self.stars = StarLayer(30000, self.rng, 1.0, band=0.3)
        self.death = self.rng.random(self.stars.n) ** 0.7
        self.base_col = self.stars.col.copy()
        self.red = blackbody(np.full(self.stars.n, 2200.0))
        self.red /= self.red.max(1, keepdims=True)

    def draw(self, t, u):
        img = canvas()
        prog = self.get('start', 0.0) + (self.get('end', 1.0) - self.get('start', 0.0)) * u
        k = np.clip((prog - self.death * 0.9) / 0.1, 0, 1)    # 0 vivo, 1 apagado
        redden = np.clip((prog - self.death * 0.9 + 0.25) / 0.25, 0, 1)
        self.stars.col = self.base_col * (1 - redden[:, None]) + self.red * redden[:, None]
        alive = (1 - k) ** 2
        cam = look_at((0, 0, 0), (math.sin(0.02 * t), 0.1, math.cos(0.02 * t)), (0, 1, 0))
        self.stars.draw(img, cam, self.get('fov', 60), t, gain=1.0, alive=alive)
        self.stars.col = self.base_col
        # resplandor débil de la radiación de Hawking muy al final
        if prog > 0.85:
            f = (prog - 0.85) / 0.15
            add_glow_point(img, W * 0.5, H * 0.45, _c('#ff9a6b', 0.6 * math.sin(f * math.pi)), 4.0)
        return img
