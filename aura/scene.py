"""Clase base de escena. Cada escena dibuja un plano en función del tiempo local."""
import numpy as np
from .core import W, H, resize


class Scene:
    #: fracción de la resolución final a la que se calcula el sombreado pesado
    scale = 1.0

    def __init__(self, seed=1, **params):
        self.seed = seed
        self.p = params
        self.rng = np.random.default_rng(seed)
        self._ready = False

    def get(self, key, default=None):
        return self.p.get(key, default)

    def prepare(self, dur):
        """Precálculos (simulaciones, nubes de estrellas). Se llama una vez."""

    def draw(self, t, u):
        """Devuelve imagen float32 (h, w, 3). t: segundos del plano, u: progreso 0..1."""
        raise NotImplementedError

    def render(self, t, dur):
        if not self._ready:
            self.prepare(dur)
            self._ready = True
        img = self.draw(t, t / max(dur, 1e-6))
        if img.shape[0] != H or img.shape[1] != W:
            img = resize(img, W, H)
        return img

    def size(self):
        return int(round(W * self.scale)), int(round(H * self.scale))
