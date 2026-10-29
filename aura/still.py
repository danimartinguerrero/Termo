"""Render de fotogramas sueltos para previsualizar escenas: python -m aura.still Escena t [k=v ...]"""
import sys, time, importlib, ast
import numpy as np, cv2
from .core import GRADES, finish

def get_scene(name):
    for mod in ('scenes_sky', 'scenes_physics', 'scenes_thermo'):
        try:
            m = importlib.import_module('aura.' + mod)
        except ModuleNotFoundError:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise KeyError(name)

def main():
    name, ts = sys.argv[1], [float(x) for x in sys.argv[2].split(',')]
    kw = {}
    grade = 'cox'; dur = 8.0; out = None
    for a in sys.argv[3:]:
        k, v = a.split('=', 1)
        try: v = ast.literal_eval(v)
        except Exception: pass
        if k == 'grade': grade = v
        elif k == 'dur': dur = v
        elif k == 'out': out = v
        else: kw[k] = v
    sc = get_scene(name)(**kw)
    tiles = []
    for t in ts:
        t0 = time.time(); img = sc.render(t, dur); t1 = time.time()
        fr = finish(img, GRADES[grade], 0)
        print(f'{name} t={t} render {t1-t0:.2f}s post {time.time()-t1:.2f}s')
        tiles.append(cv2.resize(fr, (360, 640), interpolation=cv2.INTER_AREA))
    big = np.concatenate(tiles, 1)
    cv2.imwrite(out or f'/tmp/claude-0/-home-user/e41781a3-ed51-59ea-88fb-dbb41d45a2e5/scratchpad/{name}.png', big[..., ::-1])

if __name__ == '__main__':
    main()
