"""
Montaje: línea de tiempo de planos, transiciones, efectos sincronizados con el pulso,
subtítulos/tipografía (ASS, renderizados por libass) y codificación final con ffmpeg.

Un edit se describe de forma declarativa (ver edits_*.py):

    Edit('nombre', grade='aura', music=('phonk', dict(bpm=130)),
         shots=[Shot(Nebula(palette='cliffs'), 4.0, trans='fade'), ...],
         texts=[Txt(1.0, 3.5, 'hay cien mil millones de estrellas', 'cox'), ...])
"""
import math
import os
import subprocess
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .core import W, H, FPS, GRADES, finish, canvas
from . import music as mus

FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')


# ───────────────────────────── estructuras ─────────────────────────────

@dataclass
class Shot:
    scene: object
    dur: float                    # segundos (o pulsos si el edit usa beats=True)
    trans: str = 'fade'           # transición de ENTRADA: cut | fade | flash | zoom | black | whip | glitch
    tdur: float = 0.6             # duración de la transición
    offset: float = 0.0           # tiempo inicial dentro de la escena
    speed: float = 1.0            # velocidad de reproducción de la escena
    push: float = 0.0             # zoom digital lento (Ken Burns): 0.06 = +6 % a lo largo del plano
    exposure: float = 1.0


@dataclass
class Txt:
    t0: float
    t1: float
    text: str
    style: str = 'cox'
    pos: tuple = None             # (x, y) en píxeles; None = posición del estilo
    words: bool = False           # aparición palabra a palabra
    fx: str = ''                  # etiquetas ASS adicionales


@dataclass
class Edit:
    name: str
    shots: list
    texts: list = field(default_factory=list)
    grade: str = 'cox'
    music: tuple = ('ambient', {})
    beats: bool = False           # duraciones de planos y textos expresadas en pulsos
    beat_zoom: float = 0.0        # punch-in en cada pulso tras el drop
    hit_flash: float = 0.0        # destello en golpes fuertes
    shake: float = 0.0
    letterbox: float = 0.0        # franjas negras (fracción de la altura)
    title: str = ''
    out_dir: str = 'renders'
    tail: float = 2.5             # segundos extra al final (fundido a negro)


# ───────────────────────────── estilos tipográficos ─────────────────────────────
# Colores ASS en &HAABBGGRR

STYLES = {
    # subtítulos documentales: serif fina, minúsculas, cálida (tipo Brian Cox)
    'cox': dict(file='cormorant-garamond-500-normal.ttf', font='Cormorant Garamond Medium', size=60, color='&H00E9F2F7', outline=0, shadow=2, bold=0,
                italic=0, spacing=0.5, align=2, margin_v=520, glow=None, lower=True, fade=(500, 600)),
    'cox_gold': dict(file='cormorant-garamond-600-normal.ttf', font='Cormorant Garamond SemiBold', size=60, color='&H0063CCF2', outline=0, shadow=2, bold=0,
                     italic=0, spacing=0.5, align=2, margin_v=520, glow=None, lower=True, fade=(500, 600)),
    'cox_it': dict(file='cormorant-garamond-500-italic.ttf', font='Cormorant Garamond Medium', size=64, color='&H00E9F2F7', outline=0, shadow=2, bold=0,
                   italic=1, spacing=0.5, align=5, margin_v=0, glow=None, lower=True, fade=(700, 900)),
    # título monumental romano
    'title': dict(file='cinzel-600-normal.ttf', font='Cinzel SemiBold', size=104, color='&H00FFFFFF', outline=0, shadow=0, bold=0, italic=0,
                  spacing=14, align=5, margin_v=0, glow='&H00FFD9A8', upper=True, fade=(300, 400), pop=True),
    'title_gold': dict(file='cinzel-600-normal.ttf', font='Cinzel SemiBold', size=104, color='&H0090DCFF', outline=0, shadow=0, bold=0, italic=0,
                       spacing=14, align=5, margin_v=0, glow='&H003A9CFF', upper=True, fade=(300, 400), pop=True),
    # palabra de impacto sincronizada con el pulso
    'impact': dict(file='InterDisplay-Black.otf', font='Inter Display Black', size=150, color='&H00FFFFFF', outline=0, shadow=0, bold=0, italic=0,
                   spacing=4, align=5, margin_v=0, glow='&H00FFC27A', upper=True, fade=(0, 120), pop=True),
    # ecuación elegante
    'eq': dict(file='eb-garamond-500-italic.ttf', font='EB Garamond Medium', size=96, color='&H00FFFFFF', outline=0, shadow=0, bold=0,
               italic=1, spacing=1, align=5, margin_v=0, glow='&H00FFE0B0', fade=(350, 450)),
    'eq_small': dict(file='eb-garamond-500-italic.ttf', font='EB Garamond Medium', size=70, color='&H00F2F4F6', outline=0, shadow=0, bold=0,
                     italic=1, spacing=1, align=5, margin_v=0, glow='&H00FFE0B0', fade=(300, 400)),
    # datos técnicos, estilo HUD
    'hud': dict(file='JetBrainsMono-Light.ttf', font='JetBrains Mono Light', size=36, color='&H00F0E6D8', outline=0, shadow=0, bold=0, italic=0,
                spacing=6, align=5, margin_v=0, glow='&H00FFB070', upper=True, fade=(150, 250)),
    'label': dict(file='Inter-Light.otf', font='Inter Light', size=34, color='&H00E6E6E6', outline=0, shadow=0, bold=0, italic=0,
                  spacing=16, align=5, margin_v=0, glow=None, upper=True, fade=(300, 400)),
    'sub_aura': dict(file='Inter-Medium.otf', font='Inter Medium', size=46, color='&H00FFFFFF', outline=0, shadow=1, bold=0, italic=0,
                     spacing=1, align=2, margin_v=560, glow='&H00000000', fade=(120, 200)),
    'sig': dict(file='cormorant-garamond-300-italic.ttf', font='Cormorant Garamond Light', size=44, color='&H00D8DEE6', outline=0, shadow=0, bold=0, italic=1,
                spacing=3, align=5, margin_v=0, glow=None, lower=True, fade=(800, 1200)),
}


_CMAPS = {}


def _cmap(fname):
    if fname not in _CMAPS:
        from fontTools.ttLib import TTFont
        _CMAPS[fname] = set(TTFont(os.path.join(FONTS_DIR, fname)).getBestCmap())
    return _CMAPS[fname]


def _fallback(text, style):
    """Envuelve los caracteres que la fuente del estilo no tiene con una fuente de respaldo."""
    cm = _cmap(style['file'])
    fb = 'DejaVu Serif' if 'Garamond' in style['font'] or 'Cinzel' in style['font'] else 'DejaVu Sans'
    out = []
    for ch in text:
        if ch in ' {}\\' or ord(ch) in cm:
            out.append(ch)
        else:
            out.append(f"{{\\fn{fb}\\fscx88\\fscy88}}{ch}{{\\fn{style['font']}\\fscx100\\fscy100}}")
    return ''.join(out)


def _ass_time(t):
    t = max(t, 0)
    h = int(t // 3600); mnt = int(t % 3600 // 60); s = t % 60
    return f'{h}:{mnt:02d}:{s:05.2f}'


def build_ass(texts, path):
    head = ['[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', 'WrapStyle: 0',
            'ScaledBorderAndShadow: yes', 'YCbCr Matrix: TV.709', '', '[V4+ Styles]',
            'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, '
            'Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
            'Alignment, MarginL, MarginR, MarginV, Encoding']
    for name, s in STYLES.items():
        head.append(f"Style: {name},{s['font']},{s['size']},{s['color']},&H000000FF,&H00000000,&H96000000,"
                    f"{s['bold']},{s['italic']},0,0,100,100,{s['spacing']},0,1,{s['outline']},{s['shadow']},"
                    f"{s['align']},90,90,{s['margin_v']},1")
    ev = ['', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    for tx in texts:
        s = STYLES[tx.style]
        text = tx.text
        if s.get('lower'):
            text = text.lower() if not any(c in text for c in '=∮∂γτ') else text
        if s.get('upper') and '=' not in text:
            text = text.upper()
        fi, fo = s.get('fade', (300, 300))
        tags = f'\\fad({fi},{fo})'
        if tx.pos:
            tags += f'\\pos({tx.pos[0]},{tx.pos[1]})'
        if s.get('pop'):
            tags += '\\fscx112\\fscy112\\t(0,260,0.5,\\fscx100\\fscy100)'
        tags += tx.fx
        if tx.words:
            parts = [_fallback(p, s) for p in text.split(' ')]
            dur_ms = (tx.t1 - tx.t0) * 1000
            step = min(dur_ms * 0.55 / max(len(parts), 1), 260)
            out = []
            for k, wd in enumerate(parts):
                a = int(k * step)
                out.append(f'{{\\alpha&HFF&\\t({a},{a + 220},\\alpha&H00&)}}{wd}')
            body = ' '.join(out)
        else:
            body = _fallback(text, s)
        if s.get('glow'):
            ev.append(f"Dialogue: 0,{_ass_time(tx.t0)},{_ass_time(tx.t1)},{tx.style},,0,0,0,,"
                      f"{{{tags}\\bord10\\blur18\\3c{s['glow']}\\3a&H70&\\1a&HFF&\\shad0}}{body}")
        ev.append(f"Dialogue: 1,{_ass_time(tx.t0)},{_ass_time(tx.t1)},{tx.style},,0,0,0,,{{{tags}}}{body}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(head + ev) + '\n')


# ───────────────────────────── utilidades de imagen ─────────────────────────────

def zoom_img(img, s, cx=0.5, cy=0.5, dx=0.0, dy=0.0):
    if abs(s - 1) < 1e-4 and dx == 0 and dy == 0:
        return img
    M = np.array([[s, 0, (1 - s) * W * cx + dx], [0, s, (1 - s) * H * cy + dy]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def radial_blur(img, amount, n=5):
    if amount <= 1e-3:
        return img
    acc = img.copy()
    for k in range(1, n):
        acc += zoom_img(img, 1 + amount * k / n)
    return acc / n


def motion_blur_x(img, px):
    px = int(abs(px))
    if px < 2:
        return img
    k = np.ones((1, px), np.float32) / px
    return cv2.filter2D(img, -1, k)


# ───────────────────────────── renderizador ─────────────────────────────

class Renderer:
    def __init__(self, edit):
        self.e = edit
        self.grade = GRADES[edit.grade] if isinstance(edit.grade, str) else edit.grade

    def build_music(self, dur):
        kind, kw = self.e.music
        fn = getattr(mus, kind)
        audio, info = fn(dur, **kw)
        return audio, info

    def timeline(self, beat):
        k = beat if self.e.beats else 1.0
        starts = []
        t = 0.0
        for sh in self.e.shots:
            starts.append(t)
            t += sh.dur * k
        return np.array(starts), t

    def frame(self, T, starts, total, k, info):
        e = self.e
        shots = e.shots
        idx = int(np.searchsorted(starts, T, side='right') - 1)
        idx = max(min(idx, len(shots) - 1), 0)

        def shot_img(i, Tq):
            sh = shots[i]
            dur = sh.dur * k
            tl = (Tq - starts[i]) * sh.speed + sh.offset
            img = sh.scene.render(max(tl, 0.0), dur * sh.speed + sh.offset)
            if sh.push:
                u = np.clip((Tq - starts[i]) / dur, 0, 1.2)
                img = zoom_img(img, 1 + sh.push * u)
            return img * sh.exposure

        img = shot_img(idx, T)
        extra_exp = 1.0
        extra_ca = 0.0
        # transición de entrada del plano actual (o salida hacia el siguiente)
        for j in (idx, idx + 1):
            if j <= 0 or j >= len(shots):
                continue
            sh = shots[j]
            tc = starts[j]
            td = sh.tdur
            if sh.trans == 'fade' and tc - td / 2 <= T < tc + td / 2:
                a = (T - (tc - td / 2)) / td
                a = a * a * (3 - 2 * a)
                other = shot_img(j - 1 if j == idx else j, T)
                if j == idx:
                    img = other * (1 - a) + img * a
                else:
                    img = img * (1 - a) + other * a
            elif sh.trans == 'black' and tc - td / 2 <= T < tc + td / 2:
                a = abs(T - tc) / (td / 2)
                img = img * (a ** 1.5)
            elif j == idx and T - tc < td:
                p = (T - tc) / td
                if sh.trans == 'flash':
                    extra_exp *= 1 + 5.0 * math.exp(-p * 5)
                elif sh.trans == 'zoom':
                    s = 1 + 0.22 * (1 - p) ** 3
                    img = radial_blur(zoom_img(img, s), 0.08 * (1 - p) ** 2)
                    extra_exp *= 1 + 1.2 * math.exp(-p * 6)
                elif sh.trans == 'whip':
                    off = (1 - p) ** 3 * W * 0.35
                    img = motion_blur_x(zoom_img(img, 1.0, dx=off), 120 * (1 - p) ** 2)
                elif sh.trans == 'glitch':
                    extra_ca += 0.04 * (1 - p) ** 2
                    if p < 0.4:
                        sl = np.random.default_rng(int(T * 1000)).integers(0, H - 80, 6)
                        for y in sl:
                            img[y:y + 60] = np.roll(img[y:y + 60], int(80 * (1 - p)), axis=1)
        # efectos sincronizados con el pulso
        beats = info.get('beats', np.array([]))
        drop = info.get('drop', 0.0)
        if e.beat_zoom and beats.size:
            bb = beats[(beats <= T) & (beats >= drop)]
            if bb.size:
                d = T - bb[-1]
                img = zoom_img(img, 1 + e.beat_zoom * math.exp(-d * 9))
                extra_ca += 0.006 * math.exp(-d * 10)
        hits = info.get('hits', np.array([]))
        if e.hit_flash and len(hits):
            hh = np.asarray(hits)
            hh = hh[hh <= T]
            if hh.size:
                d = T - hh[-1]
                extra_exp *= 1 + e.hit_flash * math.exp(-d * 7)
        if e.shake:
            sx = e.shake * (math.sin(T * 13.1) + 0.5 * math.sin(T * 29.7))
            sy = e.shake * (math.cos(T * 11.3) + 0.5 * math.sin(T * 23.9))
            img = zoom_img(img, 1.02, dx=sx, dy=sy)
        # fundido de entrada y salida
        fin = min(T / 0.8, 1.0)
        fout = min(max((total + e.tail - T) / 2.0, 0.0), 1.0)
        if T > total:
            fout *= max(1 - (T - total) / e.tail, 0)
        img = img * (fin * fout)
        out = finish(img, self.grade, int(T * FPS), extra_exp, extra_ca)
        if e.letterbox:
            b = int(H * e.letterbox)
            out[:b] = 0; out[-b:] = 0
        return out

    def render(self, out_path=None, preview=False, max_frames=None, start=0.0):
        e = self.e
        os.makedirs(e.out_dir, exist_ok=True)
        out_path = out_path or os.path.join(e.out_dir, f'{e.name}.mp4')
        # 1) música (necesaria antes para conocer el pulso)
        bpm_guess = e.music[1].get('bpm', 120)
        beat = 60.0 / bpm_guess
        starts, total = self.timeline(beat)
        audio, info = self.build_music(total + e.tail)
        wav = out_path[:-4] + '.wav'
        mus.write_wav(wav, audio)
        k = beat if e.beats else 1.0
        # 2) textos -> ASS (convertidos de pulsos a segundos si procede)
        texts = []
        for tx in e.texts:
            texts.append(Txt(tx.t0 * k - start, tx.t1 * k - start, tx.text, tx.style, tx.pos, tx.words, tx.fx))
        ass = out_path[:-4] + '.ass'
        build_ass(texts, ass)
        # 3) fotogramas -> ffmpeg
        nframes = int((total + e.tail) * FPS)
        if max_frames:
            nframes = min(nframes, max_frames)
        f0 = int(start * FPS)
        vf = f"ass='{ass}':fontsdir='{FONTS_DIR}'"
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
               '-i', '-', '-ss', str(start), '-i', wav, '-vf', vf, '-c:v', 'libx264', '-preset', 'slow' if not preview else 'veryfast',
               '-crf', '19' if not preview else '23', '-maxrate', '14M', '-bufsize', '28M', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-c:a', 'aac', '-b:a', '256k',
               '-shortest', '-movflags', '+faststart', out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        for f in range(f0, f0 + nframes):
            T = f / FPS
            fr = self.frame(T, starts, total, k, info)
            proc.stdin.write(fr.tobytes())
            if f % 60 == 0:
                el = time.time() - t0
                print(f'[{e.name}] {f - f0}/{nframes}  {el:.0f}s  ({el / max(f - f0, 1):.2f}s/f)', flush=True)
        proc.stdin.close()
        proc.wait()
        os.remove(wav)
        print(f'[{e.name}] listo -> {out_path} ({time.time() - t0:.0f}s)')
        return out_path
