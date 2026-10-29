"""
Edits de cosmos y astrofísica. Cada función devuelve un Edit; render.py los descubre por nombre.

Convenciones de la plantilla:
  · reflexivos  -> grade 'cox', música ambient, planos de 5–7 s, fundidos largos, subtítulos serif en minúscula
  · dinámicos   -> grade 'aura', música phonk/pulse, duraciones en pulsos (beats=True), cortes con flash/zoom
"""
from .edit import Edit, Shot, Txt
from .scenes_sky import MilkyWay, Nebula, SpiralGalaxy, AuroraSky, EarthOrbit
from .scenes_physics import (SpaceJellyfish, CosmicShower, Cherenkov, NeutrinoEarth, Pulsar, BlackHole, JWSTMirror,
                             DeepField, CMBMap, Minkowski, SpacetimeWell, HeatDeath)

F = 1.4   # duración de los fundidos reflexivos


def polvo_de_estrellas():
    """01 · Reflexivo, al estilo de los documentales de Brian Cox."""
    return Edit(
        name='01_polvo_de_estrellas', grade='cox',
        music=('ambient', dict(key='D', mode='minor', bpm=60, seed=4)),
        shots=[
            Shot(MilkyWay(seed=3, figure=True, figure_x=0.58, spin=0.006), 7.5, trans='cut', push=0.05),
            Shot(Nebula(seed=11, palette='cliffs', zoom=(1.0, 1.35)), 6.5, tdur=F),
            Shot(EarthOrbit(seed=5), 6.5, tdur=F, push=0.04),
            Shot(SpiralGalaxy(seed=2, tilt=(1.15, 0.85), dist=(3.2, 2.4)), 6.5, tdur=F),
            Shot(AuroraSky(seed=8, speed=0.7), 6.5, tdur=F, push=0.05),
            Shot(DeepField(seed=21, zoom=(1.0, 1.9)), 6.0, tdur=F),
            Shot(MilkyWay(seed=9, horizon=0.9, angle=-0.7, mw_gain=1.2), 7.0, tdur=F, push=0.06),
        ],
        texts=[
            Txt(1.2, 6.6, 'solo en nuestra galaxia hay unos cien mil millones de estrellas'),
            Txt(7.6, 12.8, 'y cada una de ellas es un horno donde se forjan los átomos'),
            Txt(14.0, 19.4, 'el hierro de tu sangre, el calcio de tus huesos'),
            Txt(20.5, 25.8, 'nacieron en el corazón de estrellas que murieron hace miles de millones de años'),
            Txt(27.0, 32.4, 'somos polvo de estrellas que ha aprendido a mirar hacia arriba', 'cox_gold'),
            Txt(33.4, 38.6, 'y cuando miramos la noche, no estamos mirando el vacío'),
            Txt(40.0, 45.0, 'estamos mirando a casa', 'cox_it'),
            Txt(43.2, 47.5, 'física · con aura', 'sig', pos=(540, 1580)),
        ])


def mensajeros_del_cosmos():
    """02 · Dinámico (phonk): rayos cósmicos, muones, neutrinos, púlsares y auroras."""
    return Edit(
        name='02_mensajeros_del_cosmos', grade='aura', beats=True,
        music=('phonk', dict(bpm=130, key='F#', seed=7, intro_bars=4)),
        beat_zoom=0.035, hit_flash=0.6,
        shots=[
            # intro (16 pulsos) contemplativa
            Shot(SpaceJellyfish(seed=4), 8, trans='cut', push=0.05),
            Shot(AuroraSky(seed=12, speed=1.3), 8, trans='fade', tdur=0.8),
            # drop
            Shot(CosmicShower(seed=5, t_hit=0.15), 8, trans='flash', tdur=0.5),
            Shot(Minkowski(beta=(0.3, 0.995)), 4, trans='zoom', tdur=0.35),
            Shot(EarthOrbit(seed=7, aurora=1.5), 4, trans='whip', tdur=0.35),
            Shot(NeutrinoEarth(seed=3), 4, trans='flash', tdur=0.4),
            Shot(Cherenkov(seed=2, t_event=0.25), 4, trans='zoom', tdur=0.35),
            Shot(Pulsar(seed=6, period=60 / 130 * 2), 8, trans='glitch', tdur=0.4),
            Shot(SpaceJellyfish(seed=9, grow_speed=1.4), 4, trans='whip', tdur=0.35, offset=2.0),
            Shot(AuroraSky(seed=31, speed=2.0, hue=0.4), 4, trans='flash', tdur=0.4),
            Shot(MilkyWay(seed=4, figure=True, spin=0.02), 8, trans='zoom', tdur=0.5, push=0.08),
        ],
        texts=[
            Txt(1.5, 7.5, 'el cielo nunca está quieto', 'cox'),
            Txt(9.0, 15.5, 'nos bombardea sin descanso', 'cox'),
            Txt(16.0, 19.8, 'rayos cósmicos', 'impact', pos=(540, 760)),
            Txt(17.0, 23.8, '3 × 10²⁰ eV', 'hud', pos=(540, 1500)),
            Txt(24.0, 27.8, 'muones', 'impact', pos=(540, 300)),
            Txt(24.3, 27.8, "τ = 2,2 μs   ·   t' = γτ", 'eq_small', pos=(540, 1560)),
            Txt(28.0, 31.8, 'auroras', 'impact', pos=(540, 1500)),
            Txt(32.0, 35.8, 'neutrinos', 'impact', pos=(540, 300)),
            Txt(32.4, 35.8, '10¹⁴ ATRAVIESAN TU CUERPO CADA SEGUNDO', 'hud', pos=(540, 1640)),
            Txt(36.0, 39.8, 'luz de cherenkov', 'title', pos=(540, 1650)),
            Txt(40.0, 47.8, 'púlsares', 'impact', pos=(540, 330)),
            Txt(41.0, 47.8, '716 VUELTAS POR SEGUNDO', 'hud', pos=(540, 1600)),
            Txt(48.0, 51.8, 'medusas espaciales', 'title', pos=(540, 1500)),
            Txt(52.0, 55.8, 'LUZ DEL VIENTO SOLAR', 'hud', pos=(540, 1540)),
            Txt(56.5, 63.5, 'y tú, en medio de todo', 'cox_it', pos=(540, 760)),
        ])


def relatividad():
    """03 · Relatividad especial y general con pulso cinematográfico."""
    bpm = 100
    b = 60 / bpm
    return Edit(
        name='03_relatividad', grade='lab', beats=True,
        music=('pulse', dict(bpm=bpm, key='D', seed=3, hits=(16 * b, 32 * b, 48 * b))),
        hit_flash=0.8, beat_zoom=0.0,
        shots=[
            Shot(Minkowski(beta=(0.0, 0.0)), 8, trans='cut'),
            Shot(Minkowski(beta=(0.0, 0.85)), 8, trans='fade', tdur=0.6),
            Shot(SpacetimeWell(seed=2, mass=(0.0, 1.8), dist=(11, 8)), 16, trans='flash', tdur=0.5),
            Shot(BlackHole(seed=4, dist=(30, 20)), 16, trans='flash', tdur=0.5),
            Shot(SpacetimeWell(seed=5, mass=(1.8, 1.8), elev=0.25, dist=(8, 6.5)), 8, trans='zoom', tdur=0.5),
            Shot(BlackHole(seed=9, dist=(18, 15), elev=(0.05, 0.08)), 8, trans='fade', tdur=0.8),
        ],
        texts=[
            Txt(0.6, 7.6, 'c = 299 792 458 m/s', 'hud', pos=(540, 330)),
            Txt(2.5, 7.6, 'la velocidad de la luz es la misma para todos', 'cox'),
            Txt(8.4, 15.8, "x' = γ (x − vt)", 'eq', pos=(540, 300)),
            Txt(9.4, 15.8, "t' = γ (t − vx/c²)", 'eq', pos=(540, 430)),
            Txt(11.0, 15.8, 'transformaciones de lorentz', 'label', pos=(540, 1650)),
            Txt(16.2, 23.5, 'la masa curva el espacio-tiempo', 'title', pos=(540, 330)),
            Txt(24.0, 31.6, 'Gμν = (8πG / c⁴) · Tμν', 'eq', pos=(540, 1560)),
            Txt(32.2, 39.5, 'y donde se curva demasiado', 'cox'),
            Txt(40.0, 47.6, 'ni la luz escapa', 'title_gold', pos=(540, 1560)),
            Txt(48.2, 55.6, 'γ = 1/√(1 − v²/c²)', 'eq', pos=(540, 330)),
            Txt(56.2, 63.8, 'el tiempo no es absoluto', 'cox_it', pos=(540, 1560)),
            Txt(60.0, 64.5, 'einstein · 1905 · 1915', 'label', pos=(540, 1700)),
        ])


def ojos_del_universo():
    """04 · Hubble, James Webb y la radiación de fondo de microondas."""
    bpm = 92
    b = 60 / bpm
    return Edit(
        name='04_ojos_del_universo', grade='aura', beats=True,
        music=('pulse', dict(bpm=bpm, key='A', seed=11, hits=(16 * b, 40 * b))),
        hit_flash=0.7,
        shots=[
            Shot(DeepField(seed=7, zoom=(1.0, 1.5)), 8, trans='cut'),
            Shot(JWSTMirror(seed=3, unfold=(0.15, 0.75)), 8, trans='fade', tdur=0.8),
            Shot(Nebula(seed=21, palette='pillars', zoom=(1.0, 1.5), slope=-0.1), 8, trans='flash', tdur=0.5),
            Shot(Nebula(seed=11, palette='cliffs', zoom=(1.1, 1.45)), 8, trans='zoom', tdur=0.4),
            Shot(DeepField(seed=33, zoom=(1.3, 2.6), focus=(0.4, 0.55)), 8, trans='whip', tdur=0.4),
            Shot(CMBMap(mode='mollweide', reveal=0.6), 12, trans='flash', tdur=0.5),
            Shot(CMBMap(mode='globe', spin=0.6), 8, trans='fade', tdur=0.8),
        ],
        texts=[
            Txt(0.8, 7.6, '25 · IV · 1990', 'hud', pos=(540, 320)),
            Txt(2.0, 7.6, 'hubble', 'title', pos=(540, 1560)),
            Txt(8.6, 15.6, 'james webb', 'title_gold', pos=(540, 330)),
            Txt(9.6, 15.6, '18 ESPEJOS DE BERILIO Y ORO · 6,5 m', 'hud', pos=(540, 1620)),
            Txt(16.2, 23.6, 'los pilares de la creación', 'cox'),
            Txt(24.2, 31.6, 'el infrarrojo atraviesa el polvo', 'cox'),
            Txt(32.2, 39.6, 'luz que salió hace 13 000 millones de años', 'cox_gold'),
            Txt(40.2, 47.6, 'fondo cósmico de microondas', 'title', pos=(540, 300)),
            Txt(41.0, 51.6, 't ≈ 380 000 años   ·   T = 2,725 K', 'hud', pos=(540, 1640)),
            Txt(52.2, 59.8, 'el eco del big bang', 'cox_it', pos=(540, 1600)),
        ])


def muerte_termica():
    """05 · Reflexivo: el destino último del universo (puente con termodinámica)."""
    return Edit(
        name='05_muerte_termica', grade='cox',
        music=('ambient', dict(key='C', mode='minor', bpm=56, seed=9, bells=True)),
        shots=[
            Shot(SpiralGalaxy(seed=8, tilt=(0.6, 0.9), dist=(2.6, 3.1), speed=0.08), 7.0, trans='cut'),
            Shot(Nebula(seed=5, palette='violet', zoom=(1.3, 1.05)), 6.5, tdur=F),
            Shot(BlackHole(seed=2, dist=(22, 26), gain=0.9), 6.5, tdur=F),
            Shot(HeatDeath(seed=3, start=0.0, end=0.55), 7.0, tdur=F),
            Shot(HeatDeath(seed=3, start=0.55, end=1.15), 9.0, tdur=F),
            Shot(MilkyWay(seed=17, figure=True, figure_x=0.42, horizon=0.86), 8.0, tdur=2.0, push=0.05),
        ],
        texts=[
            Txt(1.2, 6.4, 'todo lo que existe tiende al desorden'),
            Txt(7.6, 12.8, 'es la segunda ley de la termodinámica'),
            Txt(14.0, 19.4, 'incluso los agujeros negros acabarán evaporándose'),
            Txt(20.6, 26.2, 'las estrellas, una a una, se irán apagando'),
            Txt(27.4, 33.6, 'dentro de 10¹⁰⁰ años el universo será frío, oscuro y silencioso'),
            Txt(35.6, 40.8, 'pero eso no es ahora', 'cox_it'),
            Txt(42.0, 47.0, 'ahora las estrellas brillan, y tú estás aquí para verlas', 'cox_gold'),
            Txt(46.8, 50.8, 'física · con aura', 'sig', pos=(540, 1580)),
        ])


EDITS = {
    '01_polvo_de_estrellas': polvo_de_estrellas,
    '02_mensajeros_del_cosmos': mensajeros_del_cosmos,
    '03_relatividad': relatividad,
    '04_ojos_del_universo': ojos_del_universo,
    '05_muerte_termica': muerte_termica,
}
