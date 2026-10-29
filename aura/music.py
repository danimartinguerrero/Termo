"""
Sintetizador de bandas sonoras originales (sin samples externos):

  ambient(...)    pads cinematográficos lentos, campanas y reverberación larga (edits reflexivos)
  phonk(...)      drift phonk: cencerro melódico, 808 saturado, palmas, hi-hats (edits de aura)
  pulse(...)      pulso híbrido tipo Zimmer: ostinato, bombo, golpes 'braam' y subidas

Cada generador devuelve (audio estéreo float32 [N,2], info) donde info incluye los
tiempos de pulso ('beats'), compás ('bars') y momentos destacados ('hits', 'drop').
"""
import math
import numpy as np
from scipy import signal

SR = 44100


def note(n):
    """Número MIDI -> Hz."""
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {k: i for i, k in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}


def m(name):
    """'A3' -> 57"""
    pitch, octv = name[:-1], int(name[-1])
    return 12 * (octv + 1) + NOTE[pitch]


def env_adsr(n, a, d, s, r, sr=SR):
    a = int(a * sr); d = int(d * sr); r = int(r * sr)
    sus = max(n - a - d - r, 0)
    e = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, s, d, endpoint=False),
                        np.full(sus, s), np.linspace(s, 0, r)])
    return e[:n] if e.size >= n else np.pad(e, (0, n - e.size))


def saw(f, n, phase=0.0):
    t = np.arange(n) / SR
    # sierra con limitación de banda aproximada (suma de armónicos hasta 8 kHz)
    out = np.zeros(n)
    kmax = int(min(8000 / max(f, 1), 40))
    for k in range(1, kmax + 1):
        out += np.sin(2 * np.pi * f * k * t + phase * k) / k
    return out * 0.6


def lowpass(x, fc, q=0.707):
    b, a = signal.butter(2, min(fc / (SR / 2), 0.99), 'low')
    return signal.lfilter(b, a, x, axis=0)


def highpass(x, fc):
    b, a = signal.butter(2, min(fc / (SR / 2), 0.99), 'high')
    return signal.lfilter(b, a, x, axis=0)


def bandpass(x, f0, bw):
    lo = max(f0 - bw / 2, 20) / (SR / 2); hi = min(f0 + bw / 2, SR / 2 - 100) / (SR / 2)
    b, a = signal.butter(2, [lo, hi], 'band')
    return signal.lfilter(b, a, x, axis=0)


def reverb(x, seconds=4.0, mix=0.4, seed=0, predelay=0.02, bright=6000):
    """Reverberación por convolución con una respuesta al impulso sintética (ruido con decaimiento)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-t * 6.9 / seconds)[:, None]
    ir = lowpass(ir, bright)
    ir[:int(predelay * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:x.shape[0]] for c in range(2)], 1)
    return x * (1 - mix) + wet * mix * 1.6


def softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def pan(x, p):
    """p en [-1, 1]"""
    a = (p + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], 1)


def place(buf, x, start):
    i = int(start * SR)
    if i >= buf.shape[0]:
        return
    x = np.array(x, dtype=np.float64, copy=True)
    k = min(int(0.012 * SR), x.shape[0])
    ramp = np.linspace(1, 0, k)
    x[-k:] *= ramp[:, None] if x.ndim == 2 else ramp
    j = min(i + x.shape[0], buf.shape[0])
    buf[i:j] += x[:j - i]


def normalize(x, peak=0.89):
    x = x - x.mean(0, keepdims=True)
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def fade(x, fin=0.05, fout=1.5):
    n = x.shape[0]
    a = int(fin * SR); b = int(fout * SR)
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a)
    e[n - b:] = np.linspace(1, 0, b) ** 2
    return x * e[:, None]


# ───────────────────────────── instrumentos ─────────────────────────────

def pad_voice(midi, dur, bright=1800, detune=0.12, voices=5, seed=0, attack=1.8, release=2.5):
    rng = np.random.default_rng(seed)
    n = int((dur + release) * SR)
    f = note(midi)
    L = np.zeros(n); R = np.zeros(n)
    for v in range(voices):
        cents = (v - (voices - 1) / 2) * detune * 100 / max(voices - 1, 1) * 2
        fv = f * 2 ** (cents / 1200)
        s = saw(fv, n, rng.random() * 6.28)
        p = (v / max(voices - 1, 1)) * 2 - 1
        L += s * math.cos((p + 1) * math.pi / 4); R += s * math.sin((p + 1) * math.pi / 4)
    x = np.stack([L, R], 1) / voices
    # filtro que se abre lentamente
    x = lowpass(x, bright)
    e = env_adsr(n, attack, 1.0, 0.85, release)
    return x * e[:, None]


def bell(midi, dur=4.0, gain=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = note(midi)
    x = np.zeros(n)
    for ratio, amp, dec in ((1, 1.0, 1.6), (2.0, 0.35, 0.9), (3.01, 0.18, 0.6), (4.2, 0.08, 0.35), (5.43, 0.05, 0.25)):
        x += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / (dec * dur / 3))
    x *= np.minimum(t / 0.004, 1)
    return x * gain


def piano(midi, dur=3.0, gain=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = note(midi)
    x = np.zeros(n)
    for k in range(1, 9):
        fk = f * k * (1 + 0.0004 * k * k)
        x += np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.2 + k * 0.9)) / k ** 1.3
    x *= np.minimum(t / 0.003, 1)
    return lowpass(x, 5000) * gain


def kick(gain=1.0, punch=1.0, length=0.45, f0=150, f1=45):
    n = int(length * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 7)
    click = np.random.default_rng(1).normal(0, 1, n) * np.exp(-t * 300) * 0.3 * punch
    return softclip((x + click) * 1.5, 1.5) * gain


def snare_clap(gain=0.6, seed=3):
    rng = np.random.default_rng(seed)
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    noise = rng.normal(0, 1, n)
    env = np.zeros(n)
    for k, off in enumerate((0, 0.011, 0.022, 0.034)):
        i = int(off * SR)
        env[i:] += np.exp(-(t[:n - i]) * (60 if k < 3 else 14))
    x = bandpass(noise * env, 1400, 1800) * 1.8
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    return (x + body) * gain


def hat(gain=0.15, open_=False, seed=5):
    rng = np.random.default_rng(seed)
    n = int((0.3 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    x = highpass(rng.normal(0, 1, n), 7000) * np.exp(-t * (12 if open_ else 70))
    return x * gain


def cowbell(midi, length=0.5, gain=0.3):
    """Cencerro estilo TR-808 (dos ondas cuadradas desafinadas por un pasabanda), afinado a una nota."""
    n = int(length * SR)
    t = np.arange(n) / SR
    f = note(midi)
    sq = np.sign(np.sin(2 * np.pi * f * t)) + np.sign(np.sin(2 * np.pi * f * 1.48 * t))
    x = bandpass(sq, f * 1.6, f * 2.2)
    e = np.exp(-t * 9) * 0.8 + np.exp(-t * 40) * 0.4
    return softclip(x * e * 1.5, 1.2) * gain


def bass808(midi, length, gain=0.8, glide_from=None, drive=2.2):
    n = int(length * SR)
    t = np.arange(n) / SR
    f = note(midi) * np.ones(n)
    if glide_from is not None:
        f0 = note(glide_from)
        f = f + (f0 - note(midi)) * np.exp(-t * 18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.2 * np.sin(2 * ph)
    e = np.minimum(t / 0.005, 1) * np.exp(-t * 0.9)
    e[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return softclip(x * e * drive, drive) * gain


def riser(length, gain=0.3, seed=9):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n)
    out = np.zeros(n)
    # barrido de filtro por bloques
    blk = 2048
    for i in range(0, n, blk):
        fc = 300 + 9000 * (i / n) ** 2
        out[i:i + blk] = bandpass(x[i:i + blk], fc, fc * 0.8)
    sw = np.sin(2 * np.pi * np.cumsum(200 + 1400 * (t / t[-1]) ** 2) / SR) * 0.15
    return (out * 0.6 + sw) * (t / t[-1]) ** 2 * gain


def braam(midi, length=3.0, gain=0.6):
    """Golpe de metales graves distorsionados (estilo trailer)."""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, mm in enumerate((midi, midi + 7, midi + 12, midi - 12)):
        x += saw(note(mm) * (1 + 0.003 * k), n, k)
    x = softclip(x * 1.5, 2.5)
    x = lowpass(x, 1400)
    e = np.minimum(t / 0.02, 1) * np.exp(-t * 0.9)
    return x * e * gain


def impact(gain=0.9):
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    k = kick(1.0, 1.5, 2.5, 120, 30)
    noise = lowpass(np.random.default_rng(2).normal(0, 1, n), 2500) * np.exp(-t * 3) * 0.3
    return (k[:n] + noise) * gain


# ───────────────────────────── pistas completas ─────────────────────────────

def ambient(dur, key='D', mode='minor', bpm=64, seed=0, bells=True, swell_at=None):
    """Pads lentos con progresión emotiva, campanas dispersas y piano tenue."""
    rng = np.random.default_rng(seed)
    root = m(key + '3')
    # progresiones (grados en semitonos respecto a la tónica)
    prog = [(0, 3, 7, 10), (-4, 0, 3, 7), (-9, -5, -2, 3), (-2, 2, 5, 9)] if mode == 'minor' else \
           [(0, 4, 7, 11), (-3, 0, 4, 7), (-7, -3, 0, 4), (-5, -1, 2, 5)]
    beat = 60.0 / bpm
    bar = beat * 4
    chord_len = bar * 2
    n = int((dur + 4) * SR)
    mix = np.zeros((n, 2))
    t = 0.0; ci = 0
    while t < dur:
        ch = prog[ci % len(prog)]
        for k, iv in enumerate(ch):
            v = pad_voice(root + iv, chord_len + 0.5, bright=900 + 500 * k, seed=seed + ci * 7 + k,
                          attack=1.6, release=2.8) * 0.20
            place(mix, v, t)
        # bajo sostenido
        place(mix, pad_voice(root + ch[0] - 12, chord_len, bright=300, voices=3, seed=seed + 99 + ci) * 0.3, t)
        ci += 1
        t += chord_len
    if bells:
        scale = [0, 3, 5, 7, 10, 12, 14, 15, 19] if mode == 'minor' else [0, 2, 4, 7, 9, 12, 14, 16, 19]
        tb = bar
        while tb < dur - 2:
            mm = root + 12 + scale[rng.integers(len(scale))]
            x = bell(mm, 5.0, 0.12) if rng.random() < 0.6 else piano(mm, 4.0, 0.22)
            place(mix, pan(x, rng.uniform(-0.6, 0.6)), tb)
            tb += beat * rng.choice([1.5, 2, 3, 4])
    if swell_at is not None:
        place(mix, pan(riser(3.0, 0.18), 0), max(swell_at - 3.0, 0))
        place(mix, pan(impact(0.5), 0), swell_at)
    mix = highpass(mix, 35)
    mix = reverb(mix, 6.0, 0.55, seed, bright=5000)
    mix = fade(mix[:int(dur * SR)], 1.5, 3.0)
    beats = np.arange(0, dur, beat)
    return normalize(softclip(mix * 1.2, 1.1)), dict(bpm=bpm, beats=beats, bars=np.arange(0, dur, bar), beat=beat)


def phonk(dur, bpm=128, key='F#', seed=0, intro_bars=4, melody=None):
    """Drift phonk: intro filtrada, 'drop' con 808, cencerro melódico, palmas y hats."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    step = beat / 4
    root = m(key + '2')
    n = int((dur + 2) * SR)
    mix = np.zeros((n, 2))
    drums = np.zeros((n, 2))
    drop = intro_bars * bar
    # melodía de cencerro (16 pasos x 2 compases); None = descanso
    if melody is None:
        melody = [12, None, 12, 15, None, 12, None, 10, 12, None, 7, None, 10, None, 12, None,
                  12, None, 12, 15, None, 17, None, 15, 12, None, 10, None, 7, None, 5, None]
    bassline = [0, 0, 0, -2, -4, -4, -5, -5]       # por medio compás
    total_steps = int(dur / step) + 1
    for s in range(total_steps):
        ts = s * step
        if ts >= dur:
            break
        bar_i = int(ts / bar)
        in_drop = ts >= drop
        mel = melody[s % len(melody)]
        if mel is not None:
            g = 0.28 if in_drop else 0.2
            place(mix, pan(cowbell(root + 24 + mel, 0.4, g), -0.15), ts)
            place(mix, pan(cowbell(root + 36 + mel, 0.25, g * 0.25), 0.4), ts + step * 0.5)   # eco
        st16 = s % 16
        if in_drop:
            if st16 in (0, 6, 10) or (st16 == 14 and bar_i % 2 == 1):
                place(drums, pan(kick(0.95), 0), ts)
            if st16 in (4, 12):
                place(drums, pan(snare_clap(0.55), 0), ts)
            hg = 0.12 if st16 % 2 == 0 else 0.07
            place(drums, pan(hat(hg, seed=s), 0.25), ts)
            if st16 == 15 and bar_i % 4 == 3:
                for r in range(3):
                    place(drums, pan(hat(0.08, seed=s + r), 0.25), ts + r * step / 3)
            if st16 % 8 == 0:
                deg = bassline[(s // 8) % len(bassline)]
                prev = bassline[(s // 8 - 1) % len(bassline)]
                place(mix, pan(bass808(root + deg, step * 8, 0.75, glide_from=root + prev if deg != prev else None), 0), ts)
        else:
            if st16 in (0, 8) and bar_i >= intro_bars - 2:
                place(drums, pan(kick(0.5), 0), ts)
            if st16 % 4 == 2 and bar_i >= intro_bars - 1:
                place(drums, pan(hat(0.06, seed=s), 0.2), ts)
    # subida antes del drop e impacto
    place(mix, pan(riser(bar * 1.0, 0.25), 0), drop - bar)
    place(mix, pan(impact(0.7), 0), drop)
    # pad oscuro de fondo
    t = 0.0; k = 0
    while t < dur:
        deg = [0, -4, -5, -2][k % 4]
        for iv in (0, 3, 7):
            place(mix, pad_voice(root + 12 + deg + iv, bar * 2, bright=1200, seed=k * 3 + iv, attack=0.4, release=1.0) * 0.08, t)
        t += bar * 2; k += 1
    # intro filtrada (paso bajo) y sidechain en el drop
    idrop = int(drop * SR)
    mix[:idrop] = lowpass(mix[:idrop], 900)
    sc = np.ones(n)
    for b in np.arange(drop, dur, beat):
        i = int(b * SR); L = int(beat * SR * 0.7)
        if i + L < n:
            sc[i:i + L] = np.minimum(sc[i:i + L], 0.35 + 0.65 * np.linspace(0, 1, L) ** 0.7)
    mix = mix * sc[:, None]
    mix = reverb(mix, 2.2, 0.25, seed)
    full = softclip(mix * 0.9 + drums * 1.0, 1.4)
    full = fade(full[:int(dur * SR)], 0.02, 1.2)
    beats = np.arange(0, dur, beat)
    return normalize(full), dict(bpm=bpm, beats=beats, bars=np.arange(0, dur, bar), drop=drop, beat=beat,
                                 hits=np.arange(drop, dur, bar))


def pulse(dur, bpm=100, key='D', seed=0, hits=(), build_at=None):
    """Ostinato de cuerdas sintéticas en semicorcheas, bombo en negras, 'braams' en los golpes."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    step = beat / 4
    root = m(key + '3')
    n = int((dur + 4) * SR)
    mix = np.zeros((n, 2)); drums = np.zeros((n, 2))
    prog = [0, 0, -4, -2]
    patt = [0, 12, 7, 12, 3, 12, 7, 12]
    s = 0
    while s * step < dur:
        ts = s * step
        bi = int(ts / bar)
        deg = prog[(bi // 1) % len(prog)]
        intensity = min(ts / (bar * 4), 1.0)
        mm = root + deg + patt[s % len(patt)]
        nn = int(step * 1.2 * SR)
        tt = np.arange(nn) / SR
        x = saw(note(mm), nn) * np.exp(-tt * 14) * np.minimum(tt / 0.003, 1)
        x = lowpass(x, 1200 + 3500 * intensity)
        place(mix, pan(x * 0.16, (-0.35 if s % 2 else 0.35)), ts)
        if ts >= bar * 2 and s % 4 == 0:
            place(drums, pan(kick(0.7, 0.6), 0), ts)
        if ts >= bar * 4 and s % 8 == 4:
            place(drums, pan(snare_clap(0.25), 0), ts)
        if ts >= bar * 4:
            place(drums, pan(hat(0.05 if s % 2 else 0.08, seed=s), 0.3), ts)
        s += 1
    t = 0.0; k = 0
    while t < dur:
        deg = prog[k % 4]
        for iv in (0, 7, 12, 15):
            place(mix, pad_voice(root - 12 + deg + iv, bar, bright=1400, seed=k * 5 + iv, attack=0.8, release=1.5) * 0.09, t)
        t += bar; k += 1
    for h in hits:
        place(mix, pan(braam(root - 12 + 0, 3.0, 0.5), 0), h)
        place(drums, pan(impact(0.6), 0), h)
        place(mix, pan(riser(1.8, 0.16), 0), max(h - 1.8, 0))
    mix = reverb(mix, 3.5, 0.35, seed)
    full = softclip(mix + drums, 1.3)
    full = fade(full[:int(dur * SR)], 0.5, 2.0)
    beats = np.arange(0, dur, beat)
    return normalize(full), dict(bpm=bpm, beats=beats, bars=np.arange(0, dur, bar), beat=beat, hits=np.array(hits))


def write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
