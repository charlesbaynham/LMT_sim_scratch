# ARP frame-change finding (back-to-back pulses double-count the chirp)

**Status:** the **core/row path is now FIXED**; the **ARP composer remains
paused**. The row-based path no longer applies any inter-block frame change: at a
laser-frequency step it rebases the Bordé frame *without touching the amplitudes*
(`change_laser_frequency_in_borde_representation`), carrying the laser-detuning
integral on `AtomState` as `(t_ref, f_ref, accumulated_detuning_cycles)` and
applying it only at the lab boundary (`transform_state_vector`). The old
`_frame_change_phases` (`exp(±iπ Δf t)`) was removed from the core; a local copy
survives only inside `lmt_sim/arp.py` so the paused ARP composer keeps its old
behaviour. `lmt_sim.arp` and `tests/test_arp.py` are parked (the test module is
skipped) until the ARP composer is revisited. The note below ("No core sequence
code has been changed") describes the state *before* that fix.

## TL;DR

Modelling an ARP sweep as a staircase of short, back-to-back, fixed-detuning
sub-pulses is correct **only if you do not apply the inter-block frame change**.
The current `lmt_sim.arp.compose_arp_2x2` *does* apply it (mirroring the
row-based composer), and that double-counts the laser-frequency change, so it
converges to the wrong physics.

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

pe_fc = abs((arp.compose_arp_2x2(sp) @ [0, 1 + 0j])[0]) ** 2  # with frame change

print(pe_ode, pe_nofc, pe_fc)        # ~0.4932  0.4932  0.2784
```

## What is and isn't established

- **Solid:** for *back-to-back* sub-pulses (no free evolution between them),
  the frame change is wrong and "no frame change" is correct (matches the ODE and
  Landau–Zener). This makes the row composer give wrong results for back-to-back
  pulses with different detunings.
- **Open (your investigation):** whether the frame change is correct in its
  *intended* regime — pulses separated by **free evolution** at a fixed reference
  frequency, where there genuinely is an accumulated cross-frame offset to
  reconcile. This has not been audited here, and the core sequence code was left
  untouched. Suggested test: a `pulse(Δ₁) — freefall(τ) — pulse(Δ₂)` sequence
  compared against an independent single-fixed-frame (atomic-resonance, RWA)
  integration where the chirp lives entirely in the coupling phase.

## Corrected path (when un-paused)

Make `compose_arp_2x2` compose the per-block propagators directly (no inter-block
frame change). This still reuses the shared `_single_pulse_propagator_2x2`
primitive, so the pulse physics stays single-sourced; it simply does not use the
frame change, which is the wrong tool for a chirp. The imprinted-phase reference
across a detuning-error scan then needs an integral-of-laser-phase correction
(`exp(±i·2π·δ_err·T)`) rather than a frame change.
