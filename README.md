# Charged Particle in EM Fields

This is a part of an ongoing computational physics journal, and builds directly on the hand-written RK4 integrator from Stage 1 (numerical ODE solver project), no `scipy.integrate` or other library integrator is used here.

## Overview

A charged particle's motion under the Lorentz force,

```
m dv/dt = q(E + v x B)
dr/dt = v
```

is a 6D first-order ODE system in the state vector `[x, y, z, vx, vy, vz]`. The same RK4 stepper from Stage 1 is reused unchanged, only the field functions `E(x,y,z)` and `B(x,y,z)` passed into the derivatives function change between parts.

Three field configurations are simulated in increasing order of complexity, each checked against an analytic result before moving to the next.

## Part 1 : Uniform B field

Setup: `B = (0, 0, B0)`, no E field.

Expected: circular motion in the plane perpendicular to B (helical if
`v_parallel != 0`), at the cyclotron frequency `omega_c = qB0/m`.

**Verification (normalized units, q = m = B0 = 1):**

| Quantity | Theory | Numeric | Rel. error |
|---|---|---|---|
| Speed \|v\| | constant | drift ~5e-9 | — |
| Gyroradius `r_L = v_perp/omega_c` | 1.000000 | 1.000000 | — |
| Period `T_c = 2*pi*m/(qB0)` | 6.283185 | 6.283185 | 8e-9 |

## Part 2 : Uniform B + uniform E field (E x B drift)

Setup: adds `E = (0, E0, 0)`, `E0 = 0.2`, on top of the Part 1 B field.

Expected: guiding center drifts at `v_drift = (E x B) / B^2`, independent of the particle's charge and mass. Trajectory in the drift frame traces a cycloid.

**Verification:**

- Theory: `v_drift = (0.2, 0.0, 0.0)`
- Numeric (time-averaged velocity over 20 gyro-periods): `(0.19995, 0.00025, 0.0)`
- Absolute error: 2.5e-4

## Part 3 : Magnetic mirror (magnetic bottle)

Setup: an axisymmetric, non-uniform field with a field minimum at `z = 0`:

```
Bz(r=0, z) = B0 * (1 + (z/Lz)^2)
Br(r, z)   = -(r/2) * dBz/dz        (from div B = 0)
```

Expected: the perpendicular adiabatic invariant `mu = v_perp^2 / (2B)` is approximately conserved as the particle moves through the slowly varying field, causing it to reflect ("mirror") at the point where
`B(z_turn) = B_start * (v0/v_perp0)^2`.

**Verification:**

- Total speed conserved (E = 0 everywhere): min = max = 1.044031
- `mu`: mean = 0.5063, relative fluctuation = 1.8% (expected `mu` is an adiabatic invariant, exactly conserved only in the slowly-varying-field limit, not machine precision)
- Predicted turning point `z_turn = 1.500`; numeric max `|z|` reached = 1.386 (short of theory by an amount consistent with the `mu` fluctuation above, a real finite-gyroradius effect rather than an integration bug)
- Particle confirmed to reflect: `vz` changes sign

## Files

- `charged_particle.py` — full implementation: RK4 integrator, all three field configurations, verification checks, and plot generation
- `uniform_B.png`, `ExB_drift.png`, `magnetic_mirror.png` — trajectory and diagnostic plots for each part
