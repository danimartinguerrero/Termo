"""
Renderiza edits:  python -m aura.render <nombre|all> [--preview] [--frames N] [--start S] [--out DIR]
Ejemplo:          python -m aura.render 02_mensajeros_del_cosmos
"""
import argparse
import importlib

from .edit import Renderer

MODULES = ('aura.edits_cosmos', 'aura.edits_termo')


def catalog():
    out = {}
    for mod in MODULES:
        try:
            out.update(importlib.import_module(mod).EDITS)
        except ModuleNotFoundError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--frames', type=int, default=None)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--out', default='renders')
    a = ap.parse_args()
    cat = catalog()
    names = sorted(cat) if a.name == 'all' else [a.name]
    for n in names:
        e = cat[n]()
        e.out_dir = a.out
        Renderer(e).render(preview=a.preview, max_frames=a.frames, start=a.start)


if __name__ == '__main__':
    main()
