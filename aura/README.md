# aura — plantilla de edits de física

Motor en Python para generar **edits verticales (1080×1920, 30 fps)** de física con aura:
todos los planos son procedurales (shaders en Numba, partículas, simulaciones), la música
se sintetiza desde cero y la tipografía se compone con libass. No usa material con derechos
de autor: todo lo que ves y oyes sale de este código.

## Uso rápido

```bash
pip install numba scipy opencv-python-headless fonttools
```

```bash
python -m aura.render 02_mensajeros_del_cosmos
```

```bash
python -m aura.render all
```

Previsualizar 3 s desde el segundo 12 (calidad rápida):

```bash
python -m aura.render 03_relatividad --preview --frames 90 --start 12
```

Fotograma suelto de una escena (para ajustar parámetros):

```bash
python -m aura.still Nebula 0,6 palette="'pillars'" grade="'aura'"
```

## Cómo se monta un edit

```python
Edit(name='mi_edit', grade='aura', beats=True,                 # duraciones en pulsos
     music=('phonk', dict(bpm=130, key='F#', intro_bars=4)),   # 'ambient' | 'phonk' | 'pulse'
     beat_zoom=0.035, hit_flash=0.6,                           # efectos sincronizados con el ritmo
     shots=[Shot(SpaceJellyfish(), 8, trans='cut', push=0.05),
            Shot(CosmicShower(), 8, trans='flash')],           # cut fade flash zoom whip glitch black
     texts=[Txt(16, 20, 'rayos cósmicos', 'impact', pos=(540, 760))])
```

| Pieza | Archivo | Qué contiene |
|---|---|---|
| Núcleo | `core.py` | lienzo, ruido, puntos y líneas con antialiasing, bloom, ACES, gradación, grano, perfiles `cox`/`aura`/`lab`/`ember` |
| Cielo | `scenes_sky.py` | `MilkyWay`, `Nebula` (paletas cliffs/pillars/teal/violet), `SpiralGalaxy`, `AuroraSky`, `EarthOrbit` |
| Física | `scenes_physics.py` | `SpaceJellyfish`, `CosmicShower`, `Cherenkov`, `NeutrinoEarth`, `Pulsar`, `BlackHole`, `JWSTMirror`, `DeepField`, `CMBMap`, `Minkowski`, `SpacetimeWell`, `HeatDeath` |
| Termo | `scenes_thermo.py` | `GasParticles` (Maxwell–Boltzmann, pistón), `InkEntropy`, `HeatDiffusion`, `PVCycle` (carnot/otto/brayton), `IdealGasSurface` |
| Música | `music.py` | `ambient`, `phonk`, `pulse` + instrumentos (pads, 808, cencerro, braam, risers…) |
| Montaje | `edit.py` | línea de tiempo, transiciones, estilos tipográficos, codificación ffmpeg |
| Edits | `edits_cosmos.py`, `edits_termo.py` | los edits ya diseñados, listos para copiar y variar |

## Estilos de texto

`cox` · `cox_gold` · `cox_it` (subtítulos documentales en serif), `title` · `title_gold` (Cinzel),
`impact` (Inter Display Black), `eq` · `eq_small` (ecuaciones en EB Garamond itálica),
`hud` (JetBrains Mono), `label`, `sub_aura`, `sig` (firma final).

## Receta de estilo

- **Reflexivo** (tipo Brian Cox): grade `cox`, música `ambient`, planos de 6–7 s, fundidos de 1,4 s,
  una frase por plano, en minúscula, con aire. La última frase en `cox_it` o `cox_gold`.
- **Aura / dinámico**: grade `aura`, `phonk` a 126–132 bpm, intro contemplativa de 4 compases
  y en el drop cortes cada 4–8 pulsos con `flash`/`zoom`/`whip`; una palabra `impact` por plano
  y el dato técnico en `hud`.
- **Cinemático**: `pulse` con `hits` en los cambios de sección y `title` en cada golpe.

Las fuentes (OFL) están en `fonts/`.
