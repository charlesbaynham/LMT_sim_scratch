# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lmt-sim-scratch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Velocity slice + clearout, then square π vs ARP for the next pulse
#
# The question this notebook answers: after we **velocity-select** a cold cloud
# with a long square π pulse and clear out the rest, how well does the *next*
# (opposite-beam) LMT pulse transfer the surviving slice — and how much better is
# an **adiabatic rapid passage (ARP)** sweep than a plain square π pulse?
#
# Sequence simulated (single co/counter-propagating arm, Sr-87 clock line):
#
# 1. **Doppler distribution** of atoms at **T = 1 µK** (1-D Maxwell–Boltzmann).
# 2. **Velocity slice** — a square π pulse with **π-time 200 µs** on the **up**
#    beam (`k = +1`). Its narrow Fourier/Rabi width transfers only a thin velocity
#    class `|g, 0⟩ → |e, +1⟩`.
# 3. **Clearout** — the projective "blast" that `lmt_simulation.do_clearout`
#    implements: keep atoms found in the excited state, discard the ground-state
#    remainder. Without it the slice is pointless (the unselected atoms are still
#    there). The survivors are the velocity slice, now in `|e, +1⟩`.
# 4. **Next pulse** on the **down** beam (`k = -1`, opposite sign — this is what
#    makes the recoil *add* for LMT), driving `|e, +1⟩ → |g, +2⟩`. We compare the
#    transfer fidelity of:
#    * a **square π pulse, 45 µs**, versus
#    * an **ARP** (chirped, amplitude-shaped) adiabatic transfer.
#
# Everything uses the library's real Bordé 2×2 propagator
# (`_single_pulse_propagator_2x2`) and the staircase ARP composer
# (`lmt_sim.arp.compose_arp_2x2`); the per-velocity 2×2 picture is exact for a
# single closed two-level arm (see `arp_physics_walkthrough.ipynb`).

# %%
import sys

sys.path.insert(0, "..")

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from lmt_sim import arp
from lmt_sim import lmt_simulation as sim
from lmt_sim.lmt_simulation import (
    K_WAVEVECTOR,
    MASS_ATOM,
    RABI_FREQ,
    RECOIL_FREQUENCY_HZ,
    TRANSITION_WAVELENGTH,
)

rng = np.random.default_rng(20260626)
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 renamed trapz

GROUND = np.array([0.0, 1.0], dtype=complex)  # [c_excited, c_ground]
EXCITED = np.array([1.0, 0.0], dtype=complex)

# %% [markdown]
# ## 1. The Doppler distribution at 1 µK
#
# In 1-D the velocity spread is Maxwell–Boltzmann, $\sigma_v = \sqrt{k_B T/M}$.
# Each velocity $v$ produces a Doppler shift $v/\lambda$ on the laser, so the cloud
# has a **Doppler frequency spread** $\sigma_f = \sigma_v/\lambda$. That spread is
# what dephases pulses — and what the velocity slice narrows.

# %%
TEMPERATURE = 1e-6  # K
N_ATOMS = 6000

sigma_v = np.sqrt(constants.k * TEMPERATURE / MASS_ATOM)
v_recoil = constants.hbar * K_WAVEVECTOR / MASS_ATOM
sigma_doppler = sigma_v / TRANSITION_WAVELENGTH

print(f"T = {TEMPERATURE * 1e6:.1f} uK")
print(
    f"sigma_v        = {sigma_v * 1e3:.3f} mm/s = {sigma_v / v_recoil:.2f} recoil velocities"
)
print(f"recoil velocity= {v_recoil * 1e3:.3f} mm/s")
print(f"Doppler spread sigma_f = {sigma_doppler:.0f} Hz  (1-sigma)")

velocities = rng.normal(0.0, sigma_v, size=N_ATOMS)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(velocities * 1e3, bins=60, color="tab:blue", alpha=0.7)
ax.set_xlabel("velocity $v_z$ (mm/s)")
ax.set_ylabel("atoms")
ax.set_title(
    f"Thermal velocity distribution, {N_ATOMS} atoms at {TEMPERATURE * 1e6:.0f} µK"
)
ax.axvline(0, color="k", lw=0.8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The velocity slice: a 200 µs square π pulse
#
# A square pulse of π-time $T_\pi$ has Rabi frequency $\Omega = 1/(2T_\pi)$ and a
# resonance width set by that $\Omega$. A **long** pulse (here $T_\pi = 200\,\mu s$,
# so $\Omega = 2.5\,$kHz) is therefore **velocity-selective**: only atoms whose
# Doppler shift keeps them near resonance get transferred from $|g,0\rangle$ to
# $|e,+1\rangle$.
#
# We put the laser at the recoil-shifted resonance of the $|g,0\rangle\!\to\!|e,+1\rangle$
# pair (so the $v=0$ atom is exactly resonant); a moving atom is detuned by its
# Doppler shift and is transferred less efficiently. The library computes the full
# recoil + Doppler dependence in `_borde_frame_constants`; we just pass the atom's
# `vz`.

# %%
T_SLICE = 200e-6
RABI_SLICE = 1.0 / (2.0 * T_SLICE)  # Rabi freq for a 200 us pi pulse
DET_SLICE = arp.resonant_centre_detuning_hz(m_ground=0, k_sign=+1, vz=0.0)

print(f"slice pi-time   = {T_SLICE * 1e6:.0f} us   ->  Rabi = {RABI_SLICE:.0f} Hz")
print(
    f"slice laser detuning (v=0 resonance) = {DET_SLICE:.1f} Hz "
    f"= {DET_SLICE / RECOIL_FREQUENCY_HZ:.2f} recoil"
)


def slice_excited_population(v):
    """P(|e,+1>) after the 200 us slice pi pulse on the up beam, for velocity v."""
    U = sim._single_pulse_propagator_2x2(
        DET_SLICE, T_SLICE, RABI_SLICE, k_sign=+1, vz=v, m_ground=0
    )
    return np.abs((U @ GROUND)[0]) ** 2


# Velocity grid wide enough to contain the whole thermal cloud (sigma_v ~ 9.8 mm/s)
# AND the slice pulse's sinc sidelobes, so every exact integral below
# (survival, slice width, ensemble fidelity) is untruncated.
v_axis = np.linspace(-28e-3, 28e-3, 1401)
pe_slice = np.array([slice_excited_population(v) for v in v_axis])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(v_axis * 1e3, pe_slice, color="tab:green")
ax.set_xlabel("velocity $v_z$ (mm/s)")
ax.set_ylabel(r"$P(|e,+1\rangle)$ after slice")
ax.set_title("Velocity selectivity of the 200 µs π pulse")
ax.axvline(0, color="k", lw=0.8)
fig.tight_layout()
plt.show()

# velocity FWHM of the slice transfer
above = pe_slice >= 0.5 * pe_slice.max()
fwhm_v = (v_axis[above][-1] - v_axis[above][0]) * 1e3
print(f"slice transfer FWHM = {fwhm_v:.2f} mm/s  (peak P_e = {pe_slice.max():.3f})")

# %% [markdown]
# ## 3. The clearout: keep the slice, discard the rest
#
# `lmt_simulation.do_clearout` performs a **projective measurement**: with
# probability $P_g = |c_g|^2$ the atom is found in the ground state and is
# **discarded**; otherwise it survives, collapsed into the excited manifold and
# renormalised. Applied right after the slice, this keeps only the velocity-
# selected atoms (now in $|e,+1\rangle$) and throws the unselected cloud away.
#
# We Monte-Carlo it per atom exactly as `do_clearout` does (survive iff a uniform
# draw exceeds $P_g = 1 - P_e$).

# %%
pe_each = np.array([slice_excited_population(v) for v in velocities])
survived = rng.uniform(size=N_ATOMS) < pe_each  # survive iff NOT projected to ground
v_slice = velocities[survived]

# Exact surviving density: thermal x P_e(v) (the clearout keeps with prob P_e).
# The full thermal normalisation is the analytic Gaussian integral sqrt(2 pi) sigma_v
# (the v_axis range is wide enough that the truncation is negligible).
thermal = np.exp(-(v_axis**2) / (2 * sigma_v**2))
rho_surv = thermal * pe_slice
rho_surv_norm = rho_surv / trapz(rho_surv, v_axis)
sigma_v_slice = np.sqrt(trapz(rho_surv * v_axis**2, v_axis) / trapz(rho_surv, v_axis))
survival_fraction = trapz(rho_surv, v_axis) / (np.sqrt(2 * np.pi) * sigma_v)

# Sinc sidelobes of the square slice pulse let some far-detuned atoms survive.
sidelobe = np.abs(v_axis) > 8e-3
sidelobe_fraction = trapz(rho_surv[sidelobe], v_axis[sidelobe]) / trapz(
    rho_surv, v_axis
)

print(
    f"survival fraction         = {100 * survived.mean():.1f}%  (MC)  "
    f"/ {100 * survival_fraction:.1f}%  (exact)"
)
print(
    f"surviving slice sigma_v   = {v_slice.std() * 1e3:.3f} mm/s (MC)  "
    f"/ {sigma_v_slice * 1e3:.3f} mm/s (exact)"
)
print(
    f"narrowing factor          = {sigma_v / sigma_v_slice:.1f}x "
    f"(from {sigma_v * 1e3:.2f} -> {sigma_v_slice * 1e3:.2f} mm/s)"
)
print(
    f"survivors in sinc sidelobes (|v| > 8 mm/s) = {100 * sidelobe_fraction:.0f}% "
    f"-- a square slice pulse is not a clean filter"
)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hist(
    velocities * 1e3, bins=60, color="tab:blue", alpha=0.4, label="before (thermal)"
)
ax.hist(
    v_slice * 1e3, bins=60, color="tab:red", alpha=0.7, label="after slice + clearout"
)
ax.plot(
    v_axis * 1e3,
    rho_surv_norm * len(v_slice) * (v_axis[1] - v_axis[0]),
    color="k",
    lw=1.5,
    label="exact surviving density",
)
ax.set_xlabel("velocity $v_z$ (mm/s)")
ax.set_ylabel("atoms")
ax.set_title("Velocity slice narrows the distribution; clearout removes the rest")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The next pulse: opposite beam, `|e,+1⟩ → |g,+2⟩`
#
# In LMT you alternate beam directions so the recoil **adds**. The slice left the
# survivors in $|e,+1\rangle$; the next pulse is on the **down** beam ($k=-1$) and
# drives $|e,+1\rangle \to |g,+2\rangle$ — a further $+\hbar k$ of momentum. For the
# down-beam pulse the relevant two-level pair is $|g, m_g{=}2\rangle \leftrightarrow
# |e, m_g{-}1{=}1\rangle$, so we use `m_ground = 2, k_sign = -1` and put the laser at
# that pair's recoil-shifted resonance.
#
# Because the clearout leaves a **pure** $|e,+1\rangle$ state, any free evolution
# before this pulse is just a global phase and cannot affect the transfer
# probability — so we can compare pulses directly. The surviving slice still has a
# residual Doppler spread, which is exactly what tests a transfer pulse.

# %%
DET_NEXT = arp.resonant_centre_detuning_hz(m_ground=2, k_sign=-1, vz=0.0)
print(
    f"next-pulse laser detuning (v=0 resonance) = {DET_NEXT:.1f} Hz "
    f"= {DET_NEXT / RECOIL_FREQUENCY_HZ:.1f} recoil"
)


def transfer_fidelity_square(v, t_pulse=45e-6, rabi=RABI_FREQ):
    """P(|g,+2>) after a square pulse on the down beam, starting in |e,+1>."""
    U = sim._single_pulse_propagator_2x2(
        DET_NEXT, t_pulse, rabi, k_sign=-1, vz=v, m_ground=2
    )
    return np.abs((U @ EXCITED)[1]) ** 2


def transfer_fidelity_arp(v, T, delta_sweep_hz, omega0_factor, n=200):
    """P(|g,+2>) after an ARP sweep on the down beam, starting in |e,+1>."""
    subpulses = arp.make_arp_subpulses(
        T=T,
        delta_sweep_hz=delta_sweep_hz,
        omega0_hz=omega0_factor * RABI_FREQ,
        n=n,
        sweep_shape="tanh",
        omega_shape="sin2",
        m_ground=2,
        k_sign=-1,
        delta_centre_hz=DET_NEXT,  # fixed laser frame at the v=0 resonance
    )
    U = arp.compose_arp_2x2(subpulses, k_sign=-1, vz=v, m_ground=2)
    return np.abs((U @ EXCITED)[1]) ** 2


def ensemble_fidelity(fidelity_of_v, grid=None):
    """Slice-weighted average transfer fidelity (exact, weighted by rho_surv).

    Pass a coarser ``grid`` to speed up scans where full resolution is not needed.
    """
    if grid is None:
        grid, weight = v_axis, rho_surv
    else:
        weight = np.interp(grid, v_axis, rho_surv)
    f = np.array([fidelity_of_v(v) for v in grid])
    return trapz(weight * f, grid) / trapz(weight, grid), f


# %% [markdown]
# ## 5. Square 45 µs π pulse
#
# A 45 µs π pulse is broad (Rabi 11.1 kHz) — but the surviving slice still has
# enough residual Doppler spread that the transfer is imperfect in the wings, and
# it fails outright on the sinc-sidelobe survivors sitting tens of mm/s off
# resonance.

# %%
F_square_ens, F_square_v = ensemble_fidelity(transfer_fidelity_square)
print(
    f"<F_square (45 us)>  over the slice = {F_square_ens:.5f}   "
    f"infidelity = {1 - F_square_ens:.2e}"
)
print(
    "F_square at v = 0,1,2,3,4 mm/s:",
    [round(transfer_fidelity_square(x * 1e-3), 4) for x in range(5)],
)

# %% [markdown]
# ## 6. ARP transfer
#
# An ARP sweep (tanh detuning chirp + $\sin^2$ amplitude envelope) adiabatically
# follows the dressed state through resonance, so it inverts robustly across a band
# of detunings — provided it is given enough time to be adiabatic. We show two:
#
# * **Same duration (45 µs):** even matched to the square pulse's time, a
#   well-chosen ARP already beats it.
# * **Time-extended (450 µs):** given ~10× the time it becomes essentially perfect
#   across the whole slice.
#
# Sweep parameters: $\Delta_\mathrm{sweep} = 80\,$kHz, peak Rabi
# $\Omega_0 = 3\times$ the nominal. The 80 kHz sweep spans $\pm 40\,$kHz of detuning,
# i.e. $\pm\,40\text{kHz}\,\lambda \approx \pm 28\,$mm/s — wide enough to also catch
# the sinc-sidelobe survivors a square pulse leaves far off resonance. This breadth
# is the whole point: ARP can be made robust over a band the square π pulse simply
# cannot cover.

# %%
ARP_SWEEP_HZ = 8.0e4
ARP_OMEGA0_FAC = 3.0

F_arp45_ens, F_arp45_v = ensemble_fidelity(
    lambda v: transfer_fidelity_arp(v, 45e-6, ARP_SWEEP_HZ, ARP_OMEGA0_FAC)
)
F_arp450_ens, F_arp450_v = ensemble_fidelity(
    lambda v: transfer_fidelity_arp(v, 450e-6, ARP_SWEEP_HZ, ARP_OMEGA0_FAC)
)

print(
    f"<F_ARP  (45 us)>   over the slice = {F_arp45_ens:.5f}   "
    f"infidelity = {1 - F_arp45_ens:.2e}"
)
print(
    f"<F_ARP (450 us)>   over the slice = {F_arp450_ens:.6f}   "
    f"infidelity = {1 - F_arp450_ens:.2e}"
)

# %% [markdown]
# ## 7. Comparison
#
# The top panel overlays each pulse's transfer fidelity $F(v)$ on the surviving
# slice's velocity density. The square pulse's fidelity sags as soon as the atom is
# off-resonance; the time-extended ARP is flat at 1.0000 across the entire slice.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[2, 1])

ax1.plot(v_axis * 1e3, F_square_v, color="tab:orange", lw=2, label="square π, 45 µs")
ax1.plot(
    v_axis * 1e3,
    F_arp45_v,
    color="tab:purple",
    lw=1.5,
    ls="--",
    label="ARP, 45 µs (same time)",
)
ax1.plot(v_axis * 1e3, F_arp450_v, color="tab:green", lw=2, label="ARP, 450 µs")
ax1.fill_between(
    v_axis * 1e3,
    0,
    rho_surv_norm / rho_surv_norm.max(),
    color="tab:red",
    alpha=0.15,
    label="surviving slice (norm.)",
)
ax1.set_ylabel(r"transfer fidelity $F(v) = P(|g,+2\rangle)$")
ax1.set_ylim(0, 1.02)
ax1.set_title("Next-pulse transfer fidelity across the velocity slice")
ax1.legend(loc="lower center")

labels = ["square\n45 µs", "ARP\n45 µs", "ARP\n450 µs"]
infids = [1 - F_square_ens, 1 - F_arp45_ens, 1 - F_arp450_ens]
ax2.bar(labels, infids, color=["tab:orange", "tab:purple", "tab:green"])
ax2.set_yscale("log")
ax2.set_ylabel("ensemble infidelity\n" r"$1-\langle F\rangle$")
for i, val in enumerate(infids):
    ax2.text(i, val, f"  {val:.1e}", va="bottom", ha="center", fontsize=9)
ax2.set_title("Slice-averaged infidelity (log scale)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Robustness and convergence of the 450 µs ARP
#
# Two sanity checks that the near-unity ARP fidelity is real, not a fringe: it is
# (a) converged in the staircase resolution `n`, and (b) insensitive to ±15%
# changes in pulse time / sweep width / Rabi.

# %%
for n in (100, 200, 400):
    f, _ = ensemble_fidelity(
        lambda v: transfer_fidelity_arp(v, 450e-6, ARP_SWEEP_HZ, ARP_OMEGA0_FAC, n=n)
    )
    print(f"  n = {n:3d}:  <F> = {f:.6f}   infidelity = {1 - f:.2e}")

v_coarse = np.linspace(-28e-3, 28e-3, 351)  # coarse grid for the robustness scan
worst = 1.0
for fT in (0.85, 1.0, 1.15):
    for fd in (0.85, 1.0, 1.15):
        for fw in (0.85, 1.0, 1.15):
            f, _ = ensemble_fidelity(
                lambda v: transfer_fidelity_arp(
                    v, 450e-6 * fT, ARP_SWEEP_HZ * fd, ARP_OMEGA0_FAC * fw, n=100
                ),
                grid=v_coarse,
            )
            worst = min(worst, f)
print(f"\n450 µs ARP, worst <F> over ±15% in (T, sweep, Omega0) = {worst:.5f}")

# %% [markdown]
# ## Summary
#
# | next pulse (down beam, `|e,+1⟩ → |g,+2⟩`) | duration | ⟨F⟩ over slice | infidelity |
# |---|---|---|---|
# | square π | 45 µs | ~0.91 | ~9 × 10⁻² |
# | ARP (same time) | 45 µs | ~0.98 | ~2 × 10⁻² |
# | ARP (time-extended) | 450 µs | ~0.9999 | ~1 × 10⁻⁴ |
#
# After a 200 µs velocity slice + clearout at 1 µK, the surviving cloud still has a
# residual Doppler spread (here $\sigma_v \approx 3\,$mm/s, plus sinc-sidelobe atoms
# scattered out to tens of mm/s) that a 45 µs square π pulse cannot fully invert —
# its fidelity sags as soon as the atom is off resonance and collapses on the
# sidelobe survivors, costing ~9% per pulse. An adiabatic sweep follows the dressed
# state through resonance and is **flat at unity across the whole band**, removing
# that velocity-dependent loss almost entirely. The price is duration: matched to
# 45 µs the ARP already beats the square pulse, and given ~10× the time it reaches
# ~$10^{-4}$ infidelity (limited only by the most extreme sidelobe survivors; a
# wider sweep pushes it lower still) — the regime LMT needs, where per-pulse losses
# compound over tens of pulses.
