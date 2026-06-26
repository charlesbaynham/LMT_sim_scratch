# ARP frame-change finding (back-to-back pulses double-count the chirp)

**Status:** RESOLVED. Both the core/row path **and** the ARP composer are now
fixed. The row-based path no longer applies any inter-block frame change: at a
laser-frequency step it rebases the Bordé frame *without touching the amplitudes*
(`change_laser_frequency_in_borde_representation`), carrying the laser-detuning
integral on `AtomState` as `(t_ref, detuning_ref_hz, accumulated_detuning_cycles)` and
applying it only at the lab boundary (`transform_state_vector`). The old
`_frame_change_phases` (`exp(±iπ Δf t)`) was removed from the core.

`lmt_sim.arp.compose_arp_2x2` now composes the per-block propagators **directly**
(no inter-block frame change), so the Bordé-frame amplitudes match the row
composer exactly. The local `_frame_change_phases` copy is gone. Its
`ref_detuning_hz` argument optionally reproduces the lab-boundary laser-phase
integral `exp(±iπ (Φ − ref·T))` with `Φ = Σ_k detuning_k·dt_k`, for referencing an
imprinted phase to a fixed frame across a parameter scan. `tests/test_arp.py` is
un-parked and the three former-xfail physics tests (ODE / Landau–Zener / adiabatic
inversion / resonance symmetry) now pass. The note below ("No core sequence code
has been changed") describes the state *before* these fixes.

## TL;DR

Modelling an ARP sweep as a staircase of short, back-to-back, fixed-detuning
sub-pulses is correct **only if you do not apply the inter-block frame change**.
An early `lmt_sim.arp.compose_arp_2x2` *did* apply it (mirroring the then row-based
composer), which double-counted the laser-frequency change and converged to the
wrong physics. Both paths have since dropped the inter-block frame change (see
Status above); the table below is the diagnosis that motivated the fix.

| case (Δsweep = 4e5 Hz, Ω₀ = 11.9 kHz, T = 200 µs, linear sweep / const Ω) | P_e |
|---|---|
| Independent continuous-sweep ODE (textbook ARP Hamiltonian) | **0.4932** |
| Staircase **without** inter-block frame change | **0.4932** ✓ |
| Staircase **with** frame change (= what the row composer does) | 0.2784 ✗ |
| Analytic Landau–Zener `1 − exp(−π²Ω₀²T/Δsweep)` | 0.5028 |

The "no frame change" staircase also gives proper adiabatic inversion
(P_e = 0.998 for Δsweep = 2e4 Hz, tanh sweep + sin² envelope) and is exactly
symmetric in static detuning error. The "with frame change" version fails all of
these.

## Why (the physics)

Each sub-pulse propagator (`_single_pulse_propagator_2x2`) is built in the frame
co-rotating with **that block's** laser frequency: the coupling `omega_ab` is a
plain real number and the detuning lives in the Hamiltonian **diagonal**
(`Omega_3 ∝ 2π·detuning`). This is the standard "instantaneous-frame" ARP
description.

For a real chirp the laser is a single continuous wave: its **phase is
continuous** even though its frequency steps. Going to the instantaneous
co-rotating frame keeps the state continuous across a frequency step (the frame's
rotation *angle* `∫ω dt` is continuous; only its *rate* jumps), and adds **no**
extra term beyond the instantaneous detuning already in the diagonal. So the
correct staircase is simply the product of the per-block propagators.

The frame change `exp(∓iπ·Δf·t)` re-expresses a state from a *fixed* frame at the
old frequency to a *fixed* frame at the new frequency, correcting for the phase
the two frames accumulated **since t = 0**. Applied at a back-to-back boundary
this is wrong twice over:

1. **It double-counts the frequency.** The detuning is already in each block's
   diagonal; the frame rotation accounts for the frequency change a second time.
2. **The correct boundary correction is zero, not `Δf·t`.** Because the laser
   phase is continuous, the old and new instantaneous frames *coincide* at the
   boundary instant — they only diverge afterwards. The frame change instead
   applies the large `Δf·t_boundary` offset, a desync that never physically
   happened.

Equivalent statement: a chirp can be written either as (A) instantaneous
detuning in the diagonal + constant real coupling, or (B) fixed detuning +
time-varying coupling phase (`pulse_phase`). These are complete and equivalent;
you must pick one. The current code mixes them — diagonal detuning (A) **and** a
frame rotation (B) — which is why it is wrong.

This is exactly the limitation flagged by the `transform_state_vector` TODO
("Convert to the integral of laser phase ... not valid for time-varying laser
frequency") and by the (now superseded) CLAUDE.md guard note.

## Reproduction

```python
import numpy as np
from scipy.integrate import solve_ivp
from lmt_sim import arp
from lmt_sim import lmt_simulation as sim
from lmt_sim.lmt_simulation import RECOIL_FREQUENCY_HZ

T, omega0, ds, n = 200e-6, 1.19e4, 4e5, 8000
res = RECOIL_FREQUENCY_HZ
om_ab, drec = np.pi * omega0, 2 * np.pi * RECOIL_FREQUENCY_HZ

def rhs(t, y):                       # textbook ARP H, instantaneous frame
    Om3 = 2 * np.pi * (res + ds * (t / T - 0.5)) - drec
    return -1j * np.array([[Om3 / 2, om_ab], [om_ab, -Om3 / 2]]) @ y
pe_ode = abs(solve_ivp(rhs, [0, T], [0, 1 + 0j],
                       rtol=1e-11, atol=1e-13, max_step=T / 40000).y[0, -1]) ** 2

sp = arp.make_arp_subpulses(T=T, delta_sweep_hz=ds, omega0_hz=omega0, n=n,
                            sweep_shape="linear", omega_shape="const")

U = np.eye(2, dtype=complex)         # no frame change
for s in sp:
    U = sim._single_pulse_propagator_2x2(s.detuning_hz, s.duration, s.rabi_freq_hz,
                                         k_sign=+1, m_ground=0) @ U
pe_nofc = abs((U @ [0, 1 + 0j])[0]) ** 2

# Since the fix, arp.compose_arp_2x2(sp) == the no-frame-change product above,
# so abs((arp.compose_arp_2x2(sp) @ [0, 1 + 0j])[0])**2 now also gives ~0.4932.
print(pe_ode, pe_nofc)               # ~0.4932  0.4932
```

(Historically `compose_arp_2x2` applied the inter-block frame change and gave the
wrong `pe_fc ~ 0.2784`; that frame change has since been removed.)

## What is and isn't established

- **Solid:** for *back-to-back* sub-pulses (no free evolution between them),
  the frame change is wrong and "no frame change" is correct (matches the ODE and
  Landau–Zener). The row composer and `compose_arp_2x2` both now do this.
- **Resolved (was open):** the frame change is *not* needed in its once-intended
  regime either — pulses separated by **free evolution**. The row path carries the
  laser-detuning integral on `AtomState` and applies it only at the lab boundary
  (`transform_state_vector`), rebasing at a frequency step *without touching the
  amplitudes*. The `pulse(Δ₁) — freefall(τ) — pulse(Δ₂)` regression (a same-frequency
  rebase is an exact no-op; a two-detuning sequence matches the no-frame-change
  reference) lives in the core test suite.

## Corrected path (done)

`compose_arp_2x2` composes the per-block propagators directly (no inter-block frame
change), reusing the shared `_single_pulse_propagator_2x2` primitive so the pulse
physics stays single-sourced. The imprinted-phase reference across a detuning-error
scan is handled by the optional `ref_detuning_hz` argument, which applies the
integral-of-laser-phase `exp(±iπ (Φ − ref·T))` (`Φ = Σ_k detuning_k·dt_k`; the g–e
*relative* phase is the `exp(±i·2π·δ_err·T)` quoted earlier) rather than a frame
change. Implemented and covered by `tests/test_arp.py`.
