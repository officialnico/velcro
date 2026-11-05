"""Velcro: GPU-native global gyrokinetic turbulence code (JAX).

Electrostatic, δf, adiabatic-electron gyrokinetics on field-aligned (x, k_y, z)
grids with a Laguerre–Hermite spectral velocity space, following the design
document in ``docs/velcro_design_v3.2.md``.
"""
__version__ = "0.2.0"


def __getattr__(name):  # lazy imports keep `import velcro.basis` light
    if name == "VelcroConfig":
        from velcro.config import VelcroConfig
        return VelcroConfig
    if name == "Simulation":
        from velcro.simulation import Simulation
        return Simulation
    raise AttributeError(name)
