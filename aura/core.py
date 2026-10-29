"""
Núcleo del motor de edits: lienzo, primitivas de dibujo aditivo (puntos, líneas
con antialiasing), ruido procedural y la cadena de postprocesado cinematográfico
(bloom, tonemapping, gradación de color, viñeta, aberración cromática, grano).

Todas las imágenes son float32 (H, W, 3) en luz lineal: 0 = negro, 1 = blanco,
valores > 1 sangran en el bloom.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

W, H = 1080, 1920          # vertical 9:16
FPS = 30

cv2.setNumThreads(1)


# ───────────────────────────── utilidades ─────────────────────────────

def canvas(w=W, h=H):
    return np.zeros((h, w, 3), np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out(x, p=3):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    x = min(max(x, 0.0), 1.0)
    return x ** p


def lerp(a, b, t):
    return a + (b - a) * t


def hexcol(h, gain=1.0):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    return (c ** 2.2) * gain        # sRGB -> lineal aproximado


def resize(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC if w > img.shape[1] else cv2.INTER_AREA)


def blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def blackbody(T):
    """Color aproximado (lineal) de un cuerpo negro a temperatura T [K]."""
    T = np.asarray(T, np.float64) / 100.0
    r = np.where(T <= 66, 255.0, 329.698727446 * np.power(np.maximum(T - 60, 1e-6), -0.1332047592))
    g = np.where(T <= 66, 99.4708025861 * np.log(np.maximum(T, 1e-6)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(T - 60, 1e-6), -0.0755148492))
    b = np.where(T >= 66, 255.0, np.where(T <= 19, 0.0, 138.5177312231 * np.log(np.maximum(T - 10, 1e-6)) - 305.0447927307))
    c = np.stack([r, g, b], -1)
    c = np.clip(c, 0, 255) / 255.0
    return (c ** 2.2).astype(np.float32)


# ───────────────────────────── ruido ─────────────────────────────

@njit(inline='always', cache=True)
def _hash(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 951274213) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777215.0


@njit(inline='always', cache=True)
def vnoise3(x, y, z, seed):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z)
    fx = x - ix; fy = y - iy; fz = z - iz
    ix = int(ix); iy = int(iy); iz = int(iz)
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    a = _hash(ix, iy, iz, seed); b = _hash(ix + 1, iy, iz, seed)
    c = _hash(ix, iy + 1, iz, seed); d = _hash(ix + 1, iy + 1, iz, seed)
    e = _hash(ix, iy, iz + 1, seed); f = _hash(ix + 1, iy, iz + 1, seed)
    g = _hash(ix, iy + 1, iz + 1, seed); h = _hash(ix + 1, iy + 1, iz + 1, seed)
    k0 = a + (b - a) * ux; k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux; k3 = g + (h - g) * ux
    l0 = k0 + (k1 - k0) * uy; l1 = k2 + (k3 - k2) * uy
    return l0 + (l1 - l0) * uz


@njit(inline='always', cache=True)
def fbm3(x, y, z, octaves, seed):
    s = 0.0; amp = 0.5; norm = 0.0
    for o in range(octaves):
        s += amp * vnoise3(x, y, z, seed + o * 17)
        norm += amp
        x = x * 2.03 + 1.7; y = y * 2.03 - 3.1; z = z * 2.03 + 0.7
        amp *= 0.5
    return s / norm


@njit(inline='always', cache=True)
def ridged3(x, y, z, octaves, seed):
    s = 0.0; amp = 0.5; norm = 0.0
    for o in range(octaves):
        n = 1.0 - abs(vnoise3(x, y, z, seed + o * 31) * 2.0 - 1.0)
        s += amp * n * n
        norm += amp
        x *= 2.07; y *= 2.07; z *= 2.07
        amp *= 0.5
    return s / norm


# ───────────────────────────── dibujo aditivo ─────────────────────────────

@njit(parallel=False, cache=True)
def splat(img, xs, ys, cols, sizes):
    """Suma puntos gaussianos (estrellas, partículas). cols: (N,3) intensidad."""
    h, w, _ = img.shape
    for i in range(xs.shape[0]):
        x = xs[i]; y = ys[i]; s = sizes[i]
        if s <= 0:
            continue
        r = int(s * 3.0 + 1.5)
        cx = int(x); cy = int(y)
        if cx < -r or cx >= w + r or cy < -r or cy >= h + r:
            continue
        inv = 1.0 / (2.0 * s * s)
        norm = 1.0 / (6.2831853 * s * s) if s > 0.7 else 1.0 / (6.2831853 * 0.49)
        for yy in range(max(cy - r, 0), min(cy + r + 1, h)):
            dy = yy + 0.5 - y
            for xx in range(max(cx - r, 0), min(cx + r + 1, w)):
                dx = xx + 0.5 - x
                g = math.exp(-(dx * dx + dy * dy) * inv) * norm
                if g < 1e-4:
                    continue
                img[yy, xx, 0] += cols[i, 0] * g
                img[yy, xx, 1] += cols[i, 1] * g
                img[yy, xx, 2] += cols[i, 2] * g


@njit(cache=True)
def splat_fast(img, xs, ys, cols):
    """Puntos de 1 px con reparto bilineal (para nubes de cientos de miles)."""
    h, w, _ = img.shape
    for i in range(xs.shape[0]):
        x = xs[i] - 0.5; y = ys[i] - 0.5
        ix = int(math.floor(x)); iy = int(math.floor(y))
        if ix < 0 or iy < 0 or ix >= w - 1 or iy >= h - 1:
            continue
        fx = x - ix; fy = y - iy
        for c in range(3):
            v = cols[i, c]
            img[iy, ix, c] += v * (1 - fx) * (1 - fy)
            img[iy, ix + 1, c] += v * fx * (1 - fy)
            img[iy + 1, ix, c] += v * (1 - fx) * fy
            img[iy + 1, ix + 1, c] += v * fx * fy


@njit(cache=True)
def line(img, x0, y0, x1, y1, r, g, b, width, glow):
    """Segmento con antialiasing y halo opcional (aditivo)."""
    h, w, _ = img.shape
    pad = width + glow * 3.0 + 2.0
    minx = int(max(min(x0, x1) - pad, 0)); maxx = int(min(max(x0, x1) + pad, w - 1))
    miny = int(max(min(y0, y1) - pad, 0)); maxy = int(min(max(y0, y1) + pad, h - 1))
    dx = x1 - x0; dy = y1 - y0
    L2 = dx * dx + dy * dy
    hw = width * 0.5
    for yy in range(miny, maxy + 1):
        py = yy + 0.5
        for xx in range(minx, maxx + 1):
            px = xx + 0.5
            if L2 > 1e-9:
                t = ((px - x0) * dx + (py - y0) * dy) / L2
                t = min(max(t, 0.0), 1.0)
            else:
                t = 0.0
            qx = x0 + t * dx - px; qy = y0 + t * dy - py
            d = math.sqrt(qx * qx + qy * qy)
            a = min(max(hw + 0.5 - d, 0.0), 1.0)
            if glow > 0:
                a += 0.35 * math.exp(-d * d / (2 * glow * glow))
            if a <= 0.0005:
                continue
            img[yy, xx, 0] += r * a
            img[yy, xx, 1] += g * a
            img[yy, xx, 2] += b * a


@njit(cache=True)
def polyline(img, pts, r, g, b, width, glow):
    for i in range(pts.shape[0] - 1):
        line(img, pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1], r, g, b, width, glow)


@njit(cache=True)
def polyline_fade(img, pts, alphas, r, g, b, width, glow):
    for i in range(pts.shape[0] - 1):
        a = alphas[i]
        if a > 0.002:
            line(img, pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1], r * a, g * a, b * a, width, glow)


@njit(parallel=True, cache=True)
def disc(img, cx, cy, rad, r, g, b, soft):
    h, w, _ = img.shape
    for yy in prange(max(int(cy - rad - soft - 2), 0), min(int(cy + rad + soft + 2), h)):
        for xx in range(max(int(cx - rad - soft - 2), 0), min(int(cx + rad + soft + 2), w)):
            d = math.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
            a = min(max((rad - d) / max(soft, 0.75) + 0.5, 0.0), 1.0)
            if a > 0:
                img[yy, xx, 0] += r * a
                img[yy, xx, 1] += g * a
                img[yy, xx, 2] += b * a


def add_glow_point(img, x, y, color, radius, spikes=0, spike_len=0.0, angle=0.0, spike_w=1.2):
    """Estrella brillante con halo y picos de difracción (6+2 al estilo JWST si spikes=8)."""
    h, w, _ = img.shape
    R = int(max(radius * 6, spike_len) + 4)
    x0, x1 = max(int(x - R), 0), min(int(x + R), w)
    y0, y1 = max(int(y - R), 0), min(int(y + R), h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx + 0.5 - x; dy = yy + 0.5 - y
    d2 = dx * dx + dy * dy
    core = np.exp(-d2 / (2 * (radius * 0.35) ** 2)) * 4.0
    win = np.clip(1.0 - np.sqrt(d2) / R, 0, 1) ** 2
    halo = 1.0 / (1.0 + d2 / (radius * radius)) * 0.35 * win
    a = core + halo
    if spikes and spike_len > 0:
        if spikes == 8:
            angs = [angle + k * math.pi / 3 for k in range(6)] + [angle + math.pi / 2, angle - math.pi / 2]
            lens = [spike_len] * 6 + [spike_len * 0.55] * 2
        else:
            angs = [angle + k * 2 * math.pi / spikes for k in range(spikes)]
            lens = [spike_len] * spikes
        for an, ln in zip(angs, lens):
            ux, uy = math.cos(an), math.sin(an)
            along = dx * ux + dy * uy
            perp = np.abs(-dx * uy + dy * ux)
            m = (along > 0)
            s = np.exp(-perp ** 2 / (2 * spike_w ** 2)) * np.clip(1 - along / ln, 0, 1) ** 2.2
            a = a + np.where(m, s, 0) * 1.4
    img[y0:y1, x0:x1] += a[..., None] * color[None, None, :]


# ───────────────────────────── postprocesado ─────────────────────────────

def bloom(img, strength=0.6, threshold=0.75, radius=1.0):
    h, w, _ = img.shape
    bright = np.maximum(img - threshold, 0)
    acc = np.zeros_like(img)
    small = bright
    weights = (0.45, 0.35, 0.25, 0.2, 0.15)
    for i, wt in enumerate(weights):
        small = cv2.resize(small, (max(small.shape[1] // 2, 1), max(small.shape[0] // 2, 1)), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(small, (0, 0), 2.0 * radius)
        acc += cv2.resize(bl, (w, h), interpolation=cv2.INTER_LINEAR) * wt
    return img + acc * strength


@njit(parallel=True, cache=True)
def _finish(img, out, exposure, lift, gamma, gain, sat, vign, ca, grain, seed, contrast, tint_sh, tint_hi):
    h, w, _ = img.shape
    cx = w * 0.5; cy = h * 0.5
    inv_diag = 1.0 / math.sqrt(cx * cx + cy * cy)
    for y in prange(h):
        for x in range(w):
            dx = (x - cx) * inv_diag; dy = (y - cy) * inv_diag
            r2 = dx * dx + dy * dy
            # aberración cromática radial
            if ca > 0:
                s = ca * r2
                xr = min(max(int(x - dx * s * w), 0), w - 1); yr = min(max(int(y - dy * s * h), 0), h - 1)
                xb = min(max(int(x + dx * s * w), 0), w - 1); yb = min(max(int(y + dy * s * h), 0), h - 1)
                R = img[yr, xr, 0]; G = img[y, x, 1]; B = img[yb, xb, 2]
            else:
                R = img[y, x, 0]; G = img[y, x, 1]; B = img[y, x, 2]
            R *= exposure; G *= exposure; B *= exposure
            # tonemap ACES (Narkowicz)
            c0 = R; c1 = G; c2 = B
            R = (c0 * (2.51 * c0 + 0.03)) / (c0 * (2.43 * c0 + 0.59) + 0.14)
            G = (c1 * (2.51 * c1 + 0.03)) / (c1 * (2.43 * c1 + 0.59) + 0.14)
            B = (c2 * (2.51 * c2 + 0.03)) / (c2 * (2.43 * c2 + 0.59) + 0.14)
            R = min(max(R, 0.0), 1.0); G = min(max(G, 0.0), 1.0); B = min(max(B, 0.0), 1.0)
            # a sRGB
            R = R ** (1 / 2.2); G = G ** (1 / 2.2); B = B ** (1 / 2.2)
            # split-toning: sombras / altas luces
            L = 0.2126 * R + 0.7152 * G + 0.0722 * B
            wsh = (1 - L) * (1 - L); whi = L * L
            R += tint_sh[0] * wsh + tint_hi[0] * whi
            G += tint_sh[1] * wsh + tint_hi[1] * whi
            B += tint_sh[2] * wsh + tint_hi[2] * whi
            # saturación
            L = 0.2126 * R + 0.7152 * G + 0.0722 * B
            R = L + (R - L) * sat; G = L + (G - L) * sat; B = L + (B - L) * sat
            # contraste en S alrededor de 0.5
            R = 0.5 + (R - 0.5) * contrast; G = 0.5 + (G - 0.5) * contrast; B = 0.5 + (B - 0.5) * contrast
            # lift / gamma / gain
            R = (max(R, 0.0) * gain[0] + lift[0] * (1 - R)); G = (max(G, 0.0) * gain[1] + lift[1] * (1 - G)); B = (max(B, 0.0) * gain[2] + lift[2] * (1 - B))
            R = max(R, 0.0) ** gamma; G = max(G, 0.0) ** gamma; B = max(B, 0.0) ** gamma
            # viñeta
            v = max(1.0 - vign * (0.35 * r2 + 0.9 * r2 * r2), 0.0)
            R *= v; G *= v; B *= v
            # grano (hash por píxel y frame)
            if grain > 0:
                hsh = ((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) & 0xFFFFFF
                hsh = ((hsh ^ (hsh >> 11)) * 2654435761) & 0xFFFFFF
                n = (hsh / 16777215.0 - 0.5) * grain
                k = 1.2 - L
                R += n * k; G += n * k; B += n * k
            out[y, x, 0] = min(max(int(R * 255 + 0.5), 0), 255)
            out[y, x, 1] = min(max(int(G * 255 + 0.5), 0), 255)
            out[y, x, 2] = min(max(int(B * 255 + 0.5), 0), 255)


class Grade:
    """Perfil de gradación de color. Los edits eligen uno o lo personalizan."""

    def __init__(self, exposure=1.0, lift=(0, 0, 0), gamma=1.0, gain=(1, 1, 1), sat=1.0,
                 vignette=0.35, ca=0.0, grain=0.05, contrast=1.05,
                 shadows=(0, 0, 0), highlights=(0, 0, 0), bloom=0.6, bloom_threshold=0.7, bloom_radius=1.0):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def copy(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return Grade(**d)


GRADES = {
    # cálido y suave, tipo documental (Brian Cox / Wonders)
    'cox': Grade(exposure=1.0, lift=(0.012, 0.010, 0.016), gamma=1.0, gain=(1.03, 1.0, 0.95), sat=0.92,
                 vignette=0.55, ca=0.0015, grain=0.055, contrast=1.04,
                 shadows=(-0.01, 0.0, 0.025), highlights=(0.03, 0.012, -0.02), bloom=0.75, bloom_threshold=0.55),
    # teal & orange contrastado, para edits de aura
    'aura': Grade(exposure=1.1, lift=(0.0, 0.006, 0.012), gamma=1.02, gain=(1.03, 1.0, 1.02), sat=1.12,
                  vignette=0.6, ca=0.0035, grain=0.06, contrast=1.12,
                  shadows=(-0.02, 0.008, 0.04), highlights=(0.035, 0.012, -0.025), bloom=0.9, bloom_threshold=0.5),
    # frío y limpio, pizarra científica
    'lab': Grade(exposure=1.05, lift=(0.004, 0.006, 0.012), gamma=1.0, gain=(0.98, 1.0, 1.04), sat=1.0,
                 vignette=0.45, ca=0.002, grain=0.045, contrast=1.06,
                 shadows=(-0.01, 0.0, 0.03), highlights=(0.02, 0.01, -0.01), bloom=0.8, bloom_threshold=0.55),
    # termodinámica: ámbar / brasas sobre negro
    'ember': Grade(exposure=1.05, lift=(0.012, 0.006, 0.004), gamma=1.0, gain=(1.05, 0.99, 0.93), sat=1.05,
                   vignette=0.55, ca=0.0025, grain=0.055, contrast=1.08,
                   shadows=(0.0, -0.004, 0.02), highlights=(0.04, 0.015, -0.03), bloom=0.85, bloom_threshold=0.55),
}


def finish(img, grade, frame_idx=0, extra_exposure=1.0, extra_ca=0.0):
    if grade.bloom > 0:
        img = bloom(img, grade.bloom, grade.bloom_threshold, grade.bloom_radius)
    out = np.empty(img.shape, np.uint8)
    _finish(np.ascontiguousarray(img, dtype=np.float32), out, grade.exposure * extra_exposure,
            np.array(grade.lift, np.float32), grade.gamma, np.array(grade.gain, np.float32), grade.sat,
            grade.vignette, grade.ca + extra_ca, grade.grain, frame_idx, grade.contrast,
            np.array(grade.shadows, np.float32), np.array(grade.highlights, np.float32))
    return out


# ───────────────────────────── cámara 3D sencilla ─────────────────────────────

def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def project(P, cam, fov_deg=50, w=W, h=H):
    """Proyecta puntos (N,3) -> (x, y, z_cam). z_cam<=0 está detrás."""
    eye, r, u, f = cam
    d = P - eye
    zc = d @ f
    xc = d @ r
    yc = d @ u
    foc = 0.5 * h / math.tan(math.radians(fov_deg) / 2)
    zs = np.where(zc > 1e-4, zc, 1e-4)
    return w * 0.5 + xc / zs * foc, h * 0.5 - yc / zs * foc, zc


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
