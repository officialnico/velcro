# GKX's adiabatic-electron Cyclone case, for both codes

GKX's own benchmark of the case (`gkx_cyclone_adiabatic.toml`, unchanged):
- kinetic ions, Boltzmann electrons (τ = 1), electrostatic;
- a/L_T 2.49, a/L_n 0.8;
- GKX's example box: y0 28.2, L_x 176.25 ρ_i, jtwist 5;
- grid 64 × 64 × 24, (Nl, Nm) = (4, 8);
- run to t = 400 a/v_ti.

```
gkx examples/gkx_adiabatic/gkx_cyclone_adiabatic.toml
```

Velcro's twin, `velcro_cyclone_adiabatic.toml` (the same case through Velcro's GX-deck translation), lands in the next
commit. Once it is here:

```
python -m velcro run examples/gkx_adiabatic/velcro_cyclone_adiabatic.toml --output velcro_cyclone_adiabatic.npz
```

What to compare, and the unit conversions, are in [`../gkx_ke/README.md`](../gkx_ke/README.md): t_GX = 1.9642 t,
Q_GX = 0.3666 Q on this geometry.
