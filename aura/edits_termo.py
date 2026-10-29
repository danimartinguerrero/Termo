"""
Edits de Termodinámica (a partir de los apuntes: Tema 1, ecuaciones de Poisson de la
adiabática reversible, ciclo Joule–Brayton, ley cero, segunda ley).
"""
from .edit import Edit, Shot, Txt
from .scenes_thermo import GasParticles, InkEntropy, HeatDiffusion, PVCycle, IdealGasSurface
from .scenes_physics import HeatDeath
from .scenes_sky import SpiralGalaxy

F = 1.4


def entropia():
    """01 · Reflexivo: la entropía como flecha del tiempo."""
    return Edit(
        name='01_entropia', grade='ember',
        music=('ambient', dict(key='E', mode='minor', bpm=60, seed=21)),
        shots=[
            Shot(InkEntropy(seed=3, mode='drop', flow=0.10), 7.5, trans='cut'),
            Shot(HeatDiffusion(seed=4, diffusivity=0.2), 6.5, tdur=F),
            Shot(GasParticles(seed=2, histogram=False, n=900, init='mono'), 6.5, tdur=F),
            Shot(InkEntropy(seed=7, mode='halves', color_a='#ff9a3c', color_b='#3fb4ff', flow=0.08), 7.0, tdur=F),
            Shot(HeatDeath(seed=6, start=0.3, end=1.0), 7.0, tdur=F),
            Shot(SpiralGalaxy(seed=4, tilt=(0.9, 1.1), dist=(2.4, 3.0)), 7.0, tdur=F),
        ],
        texts=[
            Txt(1.2, 6.6, 'una gota de tinta nunca vuelve a reunirse sola'),
            Txt(8.0, 13.2, 'el calor fluye siempre de lo caliente a lo frío'),
            Txt(10.0, 13.2, 'nunca al revés', 'cox_it', pos=(540, 1210)),
            Txt(14.6, 19.8, 'S = k ln Ω', 'eq', pos=(540, 330)),
            Txt(15.2, 19.8, 'la entropía cuenta las formas de estar desordenado'),
            Txt(21.0, 26.6, 'ΔS ≥ 0', 'eq', pos=(540, 330)),
            Txt(21.6, 26.6, 'por eso recuerdas el pasado y no el futuro'),
            Txt(28.0, 33.8, 'la entropía es la flecha del tiempo', 'cox_gold'),
            Txt(35.0, 40.4, 'y el universo entero la sigue', 'cox_it'),
            Txt(38.6, 42.4, 'termodinámica · con aura', 'sig', pos=(540, 1580)),
        ])


def ciclos():
    """02 · Dinámico (phonk): ciclos de Carnot, Joule–Brayton y Otto, adiabática de Poisson."""
    return Edit(
        name='02_ciclos', grade='ember', beats=True,
        music=('phonk', dict(bpm=126, key='E', seed=13, intro_bars=4)),
        beat_zoom=0.03, hit_flash=0.5,
        shots=[
            Shot(GasParticles(seed=5, piston=(260, 60 / 126 * 4), histogram=False, n=700, v0=170, init='mb',
                              vref=420), 16, trans='cut'),
            Shot(PVCycle(cycle='carnot', trace=0.55), 8, trans='flash', tdur=0.5),
            Shot(GasParticles(seed=6, piston=(330, 60 / 126 * 2), histogram=False, n=650, v0=170, init='mb',
                              vref=420), 4, trans='zoom', tdur=0.35),
            Shot(PVCycle(cycle='brayton', trace=0.55), 8, trans='whip', tdur=0.35),
            Shot(IdealGasSurface(), 4, trans='flash', tdur=0.4),
            Shot(PVCycle(cycle='otto', trace=0.55), 8, trans='zoom', tdur=0.35),
            Shot(HeatDiffusion(seed=9, diffusivity=0.24), 4, trans='glitch', tdur=0.4),
            Shot(IdealGasSurface(), 8, trans='flash', tdur=0.5),
        ],
        texts=[
            Txt(1.0, 7.5, 'calor que se convierte en trabajo', 'cox'),
            Txt(8.5, 15.5, 'y trabajo que mueve el mundo', 'cox'),
            Txt(16.0, 23.8, 'carnot', 'title', pos=(540, 230)),
            Txt(17.0, 23.8, 'η = 1 − Tc / Th', 'eq', pos=(540, 1620)),
            Txt(24.0, 27.8, 'pVᵞ = cte', 'eq', pos=(540, 1560)),
            Txt(24.3, 27.8, 'ADIABÁTICA REVERSIBLE · POISSON', 'hud', pos=(540, 1680)),
            Txt(28.0, 35.8, 'joule – brayton', 'title_gold', pos=(540, 230)),
            Txt(29.0, 35.8, 'η = 1 − T₁ / T₂', 'eq', pos=(540, 1620)),
            Txt(36.0, 39.8, 'pV = nRT', 'impact', pos=(540, 300)),
            Txt(40.0, 47.8, 'otto', 'title', pos=(540, 230)),
            Txt(41.0, 47.8, 'W = ∮ p dV', 'eq', pos=(540, 1620)),
            Txt(48.0, 51.8, 'ley cero', 'impact', pos=(540, 960)),
            Txt(52.0, 59.5, 'dU = đQ − p dV', 'eq', pos=(540, 330)),
            Txt(53.0, 59.5, 'primera ley', 'label', pos=(540, 1650)),
        ])


def de_1e23_a_pvt():
    """03 · Del microestado al macroestado: 10²³ partículas descritas por cuatro números."""
    bpm = 96
    b = 60 / bpm
    return Edit(
        name='03_de_10e23_a_pVT', grade='ember', beats=True,
        music=('pulse', dict(bpm=bpm, key='E', seed=17, hits=(16 * b, 32 * b, 48 * b))),
        hit_flash=0.6,
        shots=[
            Shot(GasParticles(seed=8, n=900, v0=150, init='mono', histogram=True), 16, trans='cut'),
            Shot(HeatDiffusion(seed=12, diffusivity=0.18), 16, trans='flash', tdur=0.5),
            Shot(IdealGasSurface(), 16, trans='flash', tdur=0.5),
            Shot(PVCycle(cycle='carnot', trace=0.5), 16, trans='flash', tdur=0.5),
        ],
        texts=[
            Txt(0.6, 7.6, 'N ≈ 10²³ partículas', 'eq', pos=(540, 200)),
            Txt(8.0, 15.6, 'maxwell – boltzmann', 'title', pos=(540, 200)),
            Txt(9.0, 15.6, 'del caos emerge una ley', 'cox', pos=(540, 1820)),
            Txt(16.4, 23.6, 'ley cero', 'title', pos=(540, 300)),
            Txt(17.4, 31.6, 'si A ~ C y B ~ C, entonces A ~ B', 'eq_small', pos=(540, 1600)),
            Txt(24.4, 31.6, 'todo tiende al equilibrio térmico', 'cox'),
            Txt(32.4, 39.6, 'y bastan cuatro números', 'cox'),
            Txt(33.4, 47.6, 'p · V · T · n', 'eq', pos=(540, 300)),
            Txt(40.4, 47.6, 'pV = nRT', 'title_gold', pos=(540, 1600)),
            Txt(48.4, 55.6, 'ciclo de carnot', 'title', pos=(540, 230)),
            Txt(49.4, 63.6, 'el límite que ninguna máquina puede superar', 'cox', pos=(540, 1700)),
            Txt(58.0, 63.8, 'termodinámica · con aura', 'sig', pos=(540, 1820)),
        ])


EDITS = {
    '01_entropia': entropia,
    '02_ciclos': ciclos,
    '03_de_10e23_a_pVT': de_1e23_a_pvt,
}
