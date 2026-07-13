# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Realistic velocity-resolved readout of the RID 77450 symmetric MZ
#
# The leakage map in
# [`population_leakage_by_m_rid77450`](./population_leakage_by_m_rid77450.py)
# reads out populations by projecting the simulated state directly onto the
# `(m, internal)` basis — a *perfect* readout that no experiment can perform.
# This notebook simulates what we can **actually measure** in the lab and shows
# what the same final state looks like through that instrument.
#
# ## The real readout
#
# We cannot interrogate the ground state in a momentum-resolved way, but we can
# resolve the excited state. The scheme is the two-frame clock-shelving
# detection:
#
# 1. **Ground blow-away.** A 461 nm imaging flash counts — and removes — every
#    ground-state atom. It is broadband on the recoil scale, so the whole
#    ground manifold disappears as a single number with no momentum resolution.
# 2. **Clock imaging pulse.** A narrow 698 nm pulse at detuning $\delta$
#    transfers the one excited-state velocity class that is resonant,
#    $|e, m\rangle \to |g, m-1\rangle$ (up beam), back to ground. We choose it
#    as long as the original velocity-selection ("slicing") pulse — 380 µs here
#    — so its Rabi linewidth (~1.8 kHz FWHM) resolves **much less than one
#    photon recoil** (excited classes are spaced by $2\,\delta_\text{rec}
#    \approx 9.4$ kHz).
# 3. **Second imaging flash.** Counts the freshly transferred atoms: the number
#    of *excited* atoms in the addressed velocity class.
#
# One shot therefore yields the excited-state population of one velocity class;
# sweeping $\delta$ shot-to-shot builds the spectrum. Each reading shows **only
# excited-state population** — the ground-state momentum structure is invisible.
# The model lives in [`lmt_sim.readout`](../lmt_sim/readout.py); the signal is
# in units of probability per atom (multiply by the shot's atom number for
# counts).
#
# ## Axis convention: kHz, Doppler-adjusted
#
# The x-axes below are the imaging-pulse laser detuning in kHz — the knob we
# genuinely turn in the lab — but **Doppler-adjusted**: the free-fall Doppler
# ramp ($f_0 g t / c \approx 14.3$ MHz/s, evaluated at the imaging pulse
# centre) and the RID 77450 frequency anchor are already subtracted, exactly as
# `build_sequence_from_lab_pulse_dump` folds them out of the recorded pulse
# detunings. $\delta = 0$ is the bare clock transition of an atom at rest in
# the falling frame; excited class $m$ then peaks at
#
# $$\delta_\text{res}(m) = (2m - 1)\,\delta_\text{rec} + v/\lambda,$$
#
# odd multiples of the recoil frequency ($\delta_\text{rec} \approx 4.7$ kHz)
# plus the atom's residual Doppler shift. The imaging pulse's own probe shift
# ($\alpha\,\Omega^2 \approx -30$ Hz here) is included in the simulation but is
# invisible on a kHz axis. Because the ground manifold is removed before the
# pulse, each ground class is fed by exactly one excited class, so the signal
# is independent of *when* the readout fires after the sequence — the free-fall
# Doppler shift of the delay is entirely absorbed by this axis convention and
# is not something we need to worry about below.
#
# ## What is simulated
#
# The genuine RID 77450 pulse record (no clearouts, `interferometer_phase = 0`,
# see the leakage notebook for its full story), followed by the swept readout:
#
# * first for a **single atom at the slice centre** — the ideal limit in which
#   the 461 nm clearouts after velocity selection have removed everything else;
# * then for a **200 nK thermal cloud with no clearouts**, as the raw record
#   was fired — showing what the unselected atoms do to the spectrum;
# * finally the same cloud **with perfect, instantaneous clearouts** inserted
#   in every window where the sequence allows one (all intended arms
#   simultaneously excited), to quantify how much of the damage they undo.

# %%
import sys

sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm
import version_info as vs

import rid77450_pulse_record as rid77450
import lmt_sim.lmt_sequence as seq
import lmt_sim.lmt_simulation as sim
import lmt_sim.readout as readout

# %% [markdown]
# ## Build the sequence and define the imaging pulse
#
# The imaging pulse copies the slicing pulse: same beam (up, $k = +1$), same
# duration (380 µs, i.e. a π pulse at Rabi frequency $1/2T \approx 1.3$ kHz)
# and the same probe-shift coefficient — only its detuning is swept.

# %%
dump = rid77450.load_dump()
alpha, v0, _timestamps, sequence = rid77450.calibrate_and_build(dump)
print(f"probe-shift alpha = {alpha:.4g} 1/Hz   initial velocity = {v0 * 1e3:+.3f} mm/s")

slice_pulse = next(e for e in sequence if isinstance(e, seq.Pulse))
readout_duration = slice_pulse.duration
readout_rabi_hz = slice_pulse.rabi_frequency
readout_k = slice_pulse.k
readout_probe_shift_hz = alpha * readout_rabi_hz**2
assert readout_k == +1  # the slice fires on the up beam

print(
    f"imaging pulse: {readout_duration * 1e6:.0f} us pi pulse on the up beam, "
    f"Rabi {readout_rabi_hz:.0f} Hz, probe shift {readout_probe_shift_hz:+.1f} Hz"
)
print(
    f"recoil frequency {sim.RECOIL_FREQUENCY_HZ / 1e3:.2f} kHz -> excited classes "
    f"spaced by {2 * sim.RECOIL_FREQUENCY_HZ / 1e3:.2f} kHz"
)

# The sweep grid. 250 Hz steps put ~7 points across the ~1.8 kHz linewidth; the
# range covers every populated class (m = -7..+9) plus the thermal wings.
sweep_detunings_hz = np.arange(-75e3, 85e3 + 1.0, 250.0)


def sweep_readout(state, vz):
    """Sweep the imaging pulse over one atom's final state."""
    return readout.simulate_excited_state_readout(
        state,
        sweep_detunings_hz,
        pulse_duration=readout_duration,
        pulse_rabi_frequency=readout_rabi_hz,
        k_sign=readout_k,
        vz=vz,
        probe_shift_coefficient=alpha,
    )


def perfect_excited_populations(state):
    """The perfect projective readout: coherent P_e(m), as the leakage map uses."""
    merged = readout.merge_rows_by_momentum_class(readout.remove_ground_rows(state))
    return {
        int(m): float(np.abs(a) ** 2)
        for m, a in zip(merged.m_values, merged.amplitudes)
    }


def class_resonances_hz(m_classes, atom_v0=0.0):
    """Doppler-adjusted detuning at which each excited class peaks (probe shift in)."""
    return (
        readout.readout_resonance_detuning_hz(
            np.asarray(m_classes), k_sign=readout_k, v0=atom_v0
        )
        + readout_probe_shift_hz
    )


# %% [markdown]
# ## A single atom from the selected slice (ideal-clearout limit)
#
# One atom at the centre of the sliced velocity class (v = 0 in the
# freely-falling simulation frame — the build folded the fitted launch velocity
# into the detunings). This is what the swept readout would show if the 461 nm
# clearouts after velocity selection had removed every unselected atom.

# %%
initial = sim.make_atom_states(c0=1, c1=0, initial_velocity_z=0.0)
final_state, _, _ = seq.run_pulse_sequence_in_borde_representation(
    initial, sequence, initial_velocity_z=0.0, discard_threshold=1e-9
)

single_perfect = perfect_excited_populations(final_state)
single_m, single_per_class = sweep_readout(final_state, vz=0.0)
single_total = single_per_class.sum(axis=1)

print("perfect P_e(m):", {m: round(p, 5) for m, p in sorted(single_perfect.items())})
for j, m in enumerate(single_m):
    res = class_resonances_hz(m)
    i = np.argmin(np.abs(sweep_detunings_hz - res))
    print(
        f"  m={m:+d}: resonance {res / 1e3:+6.2f} kHz, swept signal there "
        f"{single_total[i]:.5f} vs perfect {single_perfect[int(m)]:.5f}"
    )

# Measured linewidth of the biggest peak, to put "sub-recoil" in numbers.
res_1 = float(class_resonances_hz(1))
near = np.abs(sweep_detunings_hz - res_1) < 4e3
above_half = sweep_detunings_hz[near][single_total[near] > single_total[near].max() / 2]
fwhm_hz = above_half.max() - above_half.min()
print(
    f"m=+1 peak FWHM = {fwhm_hz / 1e3:.2f} kHz "
    f"({fwhm_hz / (2 * sim.RECOIL_FREQUENCY_HZ):.2f} of the class spacing)"
)


# %%
def plot_swept_readout(
    curves,
    perfect_stems,
    title,
    *,
    log_floor=1e-6,
    stem_label="perfect projective readout $P_e(m)$",
):
    """Linear + log panels of swept readout curves vs the perfect projection.

    ``curves`` is a list of ``(signal_array, style_kwargs)`` drawn on both
    panels; ``perfect_stems`` is a ``{m: population}`` dict drawn as stems at
    each class's resonance detuning.
    """
    fig, (ax_lin, ax_log) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    x = sweep_detunings_hz / 1e3
    for signal, style in curves:
        ax_lin.plot(x, signal, **style)
        ax_log.semilogy(x, np.maximum(signal, log_floor), **style)

    stem_m = sorted(perfect_stems)
    stem_x = class_resonances_hz(stem_m) / 1e3
    stem_y = np.array([perfect_stems[m] for m in stem_m])
    for ax in (ax_lin, ax_log):
        bottom = 0.0 if ax is ax_lin else log_floor
        ax.vlines(stem_x, bottom, np.maximum(stem_y, bottom), color="k", lw=1.0)
        shown = stem_y > bottom
        ax.plot(
            stem_x[shown],
            stem_y[shown],
            "o",
            mfc="none",
            mec="k",
            ms=6,
            label=stem_label,
        )
        # Label every class position along the top of the log panel (which has
        # no legend to collide with), populated or not -- the empty rungs are
        # part of the story.
        if ax is ax_log:
            for m, xm in zip(stem_m, stem_x):
                ax.annotate(
                    f"$m={m:+d}$",
                    (xm, 0.98),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="0.35",
                )
        ax.grid(alpha=0.25)
        ax.set_ylabel("detected fraction per shot")
    ax_log.set_ylim(log_floor, 1.5)
    ax_log.set_xlabel("imaging-pulse detuning (kHz, Doppler-adjusted)")
    ax_lin.legend(loc="upper left")
    ax_lin.set_title(title)
    vs.tag_plot(ax=ax_lin, small=True)
    return fig, (ax_lin, ax_log)


plot_swept_readout(
    [
        (
            single_total,
            dict(color="tab:red", lw=1.4, label="swept clock imaging pulse"),
        )
    ],
    single_perfect,
    "Swept excited-state readout, single atom from the selected slice "
    "(RID 77450, phase = 0)",
)
plt.show()

# %% [markdown]
# ### Reading the single-atom sweep
#
# * Two excited classes dominate: **m = +1 at +4.7 kHz (0.50) and m = +3 at
#   +23.5 kHz (0.28)** (odd multiples of the recoil frequency). The swept peak
#   heights land on the perfect projective populations (circles) to a few
#   parts in $10^3$ — on resonance the π imaging pulse transfers the whole
#   class. (Spoiler from the clearout section below: the intended φ = 0 output
#   of this sequence is m = +1 excited plus m = +2 ground *only* — the large
#   m = +3 peak is parasitic-path interference, not a second output port.)
# * The linewidth (~1.8 kHz FWHM) is a fifth of the 9.4 kHz class spacing: the
#   readout genuinely resolves individual recoil rungs, which is the point of
#   matching the slicing-pulse duration.
# * The **square imaging pulse has sinc sidelobes**: the oscillatory floor at
#   $10^{-3}$–$10^{-4}$ across the whole sweep is off-resonant transfer from
#   the two big classes, not population. The weakest leakage classes (m = −3 at
#   $6\times10^{-4}$, m = +5 at $7\times10^{-4}$) sit right at that floor:
#   their peaks ride on sidelobe background of the same size, and classes below
#   $\sim10^{-4}$ (m = −5, +7) are buried entirely. A perfect readout sees
#   them; this instrument cannot, unless the imaging pulse is shaped to kill
#   its sidelobes.
# * $P_e(m=-1) \approx 1.3\times10^{-2}$ — the largest leakage class from the
#   mirror train — is comfortably measurable at −14.2 kHz.

# %% [markdown]
# ## The whole cloud: 200 nK, no clearouts
#
# The record was fired with **no 461 nm clearouts**, so in a real shot the
# *entire* thermal cloud — not just the 380 µs-selected slice — experiences all
# 20 pulses and contributes to the excited state. We model a 200 nK cloud (the
# temperature used across the other notebooks) as a deterministic grid of
# velocity classes with Maxwell-Boltzmann weights: each atom is one full
# quantum run, so a weighted grid gives the smooth ensemble average without
# Monte-Carlo shot noise. The slice pulse selects atoms near v = 0; atoms in
# the wings are far off resonance from it but are still shaken hard by the
# 56-67 µs π pulses of the launch/mirror train (Rabi ~7-9 kHz, broader than the
# thermal Doppler spread), ending up substantially excited.
#
# The discard threshold is raised to $10^{-7}$ here (vs $10^{-9}$ above) to
# keep the 41-atom ensemble tractable; it perturbs the per-class populations at
# the percent level, well below anything visible on these axes.

# %%
TEMPERATURE = 200e-9
sigma_v = np.sqrt(constants.k * TEMPERATURE / sim.MASS_ATOM)
N_VELOCITIES = 41
velocity_grid = np.linspace(-3, 3, N_VELOCITIES) * sigma_v
velocity_weights = np.exp(-0.5 * (velocity_grid / sigma_v) ** 2)
velocity_weights /= velocity_weights.sum()

print(
    f"sigma_v = {sigma_v * 1e3:.2f} mm/s -> Doppler sigma "
    f"{sigma_v / sim.TRANSITION_WAVELENGTH / 1e3:.1f} kHz "
    f"(vs {2 * sim.RECOIL_FREQUENCY_HZ / 1e3:.2f} kHz class spacing)"
)

cloud_signal_by_class = {}  # m -> ensemble-weighted swept signal
cloud_perfect = {}  # m -> ensemble-weighted perfect P_e(m)
for v, weight in tqdm(list(zip(velocity_grid, velocity_weights)), desc="Cloud readout"):
    initial = sim.make_atom_states(c0=1, c1=0, initial_velocity_z=v)
    state, _, _ = seq.run_pulse_sequence_in_borde_representation(
        initial, sequence, initial_velocity_z=v, discard_threshold=1e-7
    )
    m_classes, per_class = sweep_readout(state, vz=v)
    for j, m in enumerate(m_classes):
        m = int(m)
        cloud_signal_by_class[m] = (
            cloud_signal_by_class.get(m, 0.0) + weight * per_class[:, j]
        )
    for m, p in perfect_excited_populations(state).items():
        cloud_perfect[m] = cloud_perfect.get(m, 0.0) + weight * p

cloud_total = np.sum(list(cloud_signal_by_class.values()), axis=0)
print(
    f"classes populated: {sorted(cloud_signal_by_class)}; "
    f"total excited fraction {sum(cloud_perfect.values()):.3f}"
)
print("per-class ensemble populations and swept peaks:")
for m in sorted(cloud_perfect, key=lambda m: -cloud_perfect[m]):
    i_peak = int(np.argmax(cloud_signal_by_class[m]))
    print(
        f"  m={m:+d}: total P_e = {cloud_perfect[m]:.4f}; swept curve peaks at "
        f"{sweep_detunings_hz[i_peak] / 1e3:+6.2f} kHz with {cloud_signal_by_class[m][i_peak]:.4f} "
        f"(cold-atom resonance {float(class_resonances_hz(m)) / 1e3:+6.2f} kHz)"
    )

# %%
plot_swept_readout(
    [
        (
            single_total,
            dict(
                color="0.6",
                lw=1.0,
                label="selected atom only (ideal clearouts)",
            ),
        ),
        (
            cloud_total,
            dict(color="tab:red", lw=1.4, label="full 200 nK cloud, no clearouts"),
        ),
    ],
    cloud_perfect,
    "Swept excited-state readout of the whole cloud (RID 77450, phase = 0, "
    "no clearouts)",
)
plt.show()

# %% [markdown]
# ### Which class produces which feature
#
# The same cloud sweep, decomposed by the excited momentum class the signal
# came from (exact: with the ground manifold blown away, ground class $m-1$ is
# fed only by excited class $m$). Each class keeps a fixed colour and is
# labelled where its own curve peaks — which for the Doppler-broadened classes
# is *not* the cold-atom rung position: e.g. the faint m = −5 and m = +7
# reservoirs live entirely in the thermal wings, so their humps sit several
# kHz from where a cold atom would resonate.

# %%
class_list = sorted(cloud_signal_by_class)
class_colors = {m: plt.get_cmap("tab10")(j % 10) for j, m in enumerate(class_list)}

fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
x = sweep_detunings_hz / 1e3
LOG_FLOOR = 1e-6
for m in class_list:
    ax.semilogy(
        x,
        np.maximum(cloud_signal_by_class[m], LOG_FLOOR),
        color=class_colors[m],
        lw=1.1,
    )
    # Label each class where its own curve actually peaks -- for the broadened
    # classes that is NOT the cold-atom resonance position.
    i_peak = int(np.argmax(cloud_signal_by_class[m]))
    ax.annotate(
        f"$m={m:+d}$",
        (x[i_peak], cloud_signal_by_class[m][i_peak] * 1.4),
        ha="center",
        fontsize=9,
        color=class_colors[m],
    )
ax.semilogy(x, np.maximum(cloud_total, LOG_FLOOR), color="k", lw=1.6, alpha=0.75)
ax.annotate("total", (x[-1], cloud_total[-1] * 1.5), ha="right", color="k")
ax.set_ylim(LOG_FLOOR, 1.0)
ax.set_xlabel("imaging-pulse detuning (kHz, Doppler-adjusted)")
ax.set_ylabel("detected fraction per shot")
ax.grid(alpha=0.25)
ax.set_title(
    "Cloud readout decomposed by excited momentum class "
    "(each class Doppler-broadened by the unselected atoms)"
)
vs.tag_plot(ax=ax, small=True)
plt.show()

# %% [markdown]
# ### What the real readout would show us
#
# * **The interferometer peaks survive, with reduced contrast.** The selected
#   slice still produces its sharp sub-recoil peaks at +4.7 kHz (m = +1) and
#   +23.5 kHz (m = +3), essentially at the single-atom positions (the maxima
#   are pulled a few hundred Hz by the asymmetric pedestal underneath) — but
#   scaled down by the small fraction of the cloud the 380 µs slice picked
#   out, and now sitting on a comparable background.
# * **The unselected cloud is far from spectator.** With no clearouts, half the
#   *whole* cloud ends the sequence excited: the 56-67 µs π pulses (Rabi
#   7-9 kHz) are broader than the thermal Doppler spread and shake every atom.
#   Strikingly, the biggest single reservoir is **m = −1, holding as much
#   population as the m = +1 port** (~0.2 each) — wing atoms that the down-beam
#   pulses drive one recoil the *wrong* way. Each class is smeared into a
#   Doppler pedestal (~6 kHz sigma) by the wings' velocities; the pedestals
#   overlap, fill the gaps between rungs, and bury the $10^{-3}$-level leakage
#   peaks of the single-atom picture. **This is the quantitative case for the
#   461 nm clearouts** in the real sequence: removing the unselected ground
#   atoms right after the slice pushes the background towards the grey
#   ideal-clearout curve.
# * **The stems and the swept curve disagree on purpose.** The circles are each
#   class's *total* excited population; a swept shot only recovers the part of
#   a class that is resonant at that detuning. The cold selected slice is
#   unbroadened, so its peaks reach their stems; the Doppler-broadened cloud
#   classes fall an order of magnitude short (m = −1: stem at 0.21, swept
#   maximum ~0.05) because their population is spread across the pedestal. A
#   velocity-resolved readout necessarily reports class populations only after
#   *integrating* over the pedestal.
# * **Instrument response matters at the $10^{-3}$ level.** Even with perfect
#   clearouts, the square imaging pulse's sinc sidelobes put a $10^{-3}$ to
#   $10^{-4}$ oscillatory floor everywhere, which masks the weakest leakage
#   classes. Shaping the imaging pulse (or fitting sidelobe-aware models to the
#   sweep) would be needed to see them.
# * Everything above is per shot and per atom: one detuning per experimental
#   run, so a full sweep at this 250 Hz resolution is ~640 shots — the
#   simulation is telling us where to spend them.

# %% [markdown]
# ## Adding the clearouts
#
# A 461 nm clearout removes every ground-state atom, so it can only fire when
# **everything the sequence means to keep is in the excited state** — for the
# interferometer proper, when *both* arms are simultaneously excited (before
# the first beamsplitter there is a single cloud, so the classic post-slice
# window counts too). Walking the intended arm trajectories through the
# recorded pulses finds every such window; per the sequence structure they
# fall in the free-fall gaps after pulses **0, 2, 6, 11 and 15**. We insert a
# clearout in *all* of them, assumed **instantaneous and perfect** (every
# ground atom removed, the excited amplitudes untouched).
#
# The model (`readout.run_sequence_with_ground_clearouts`) is deterministic
# and ensemble-averaged, unlike the Monte-Carlo `do_clearout`: at each window
# the ground rows are dropped and their weight is accumulated into a single
# per-atom survival factor, so the final readout stays in absolute
# atoms-per-original-atom units. (The sequence machinery itself is untouched —
# the run is split into segments at the clearout instants; because the
# post-clearout state is entirely excited, the segment restart is provably a
# global phase and the populations are exact. See the tests.)

# %%
clearout_windows = rid77450.both_arms_excited_after_pulses(dump)
arms = rid77450.intended_arm_trajectories(dump, force_bs_pi2=False)
print(f"clearout windows: after pulses {clearout_windows}")
for p in clearout_windows:
    states = sorted({hist[p] for hist, _cid in arms if p in hist})
    labels = ", ".join(f"|e, m={m:+d}>" for m, _g in states)
    print(f"  after pulse {p:2d}: intended arms in {labels}")

# %% [markdown]
# ### The single sliced atom with clearouts
#
# The ideal arm walk says these windows are clean — but the walk assumes
# perfect π pulses. The genuine state carries imperfect-transfer residue in
# the ground manifold at each window, and the clearouts remove it for good
# (instead of letting later resonant pulses coherently re-absorb it into the
# arms). The survival product tells us what that costs even the
# perfectly-selected atom; the sweep shows what it does to the spectrum.

# %%
single_co_state, single_co_weight = readout.run_sequence_with_ground_clearouts(
    sequence,
    clearout_windows,
    initial_velocity_z=0.0,
    discard_threshold=1e-9,
)
print(f"sliced atom survives all clearouts with weight {single_co_weight:.4f}")

single_co_perfect = {
    m: single_co_weight * p
    for m, p in perfect_excited_populations(single_co_state).items()
}
single_co_m, single_co_per_class = sweep_readout(single_co_state, vz=0.0)
single_co_total = single_co_weight * single_co_per_class.sum(axis=1)
print(
    "absolute P_e(m) with clearouts:",
    {m: round(p, 5) for m, p in sorted(single_co_perfect.items())},
)

# %% [markdown]
# ### The whole cloud with clearouts

# %%
cloud_co_signal_by_class = {}
cloud_co_perfect = {}
cloud_co_survival = 0.0
for v, w_v in tqdm(
    list(zip(velocity_grid, velocity_weights)), desc="Cloud + clearouts"
):
    state, w_atom = readout.run_sequence_with_ground_clearouts(
        sequence,
        clearout_windows,
        initial_velocity_z=v,
        discard_threshold=1e-7,
    )
    if state is None:
        continue
    cloud_co_survival += w_v * w_atom
    m_classes, per_class = sweep_readout(state, vz=v)
    for j, m in enumerate(m_classes):
        m = int(m)
        cloud_co_signal_by_class[m] = (
            cloud_co_signal_by_class.get(m, 0.0) + w_v * w_atom * per_class[:, j]
        )
    for m, p in perfect_excited_populations(state).items():
        cloud_co_perfect[m] = cloud_co_perfect.get(m, 0.0) + w_v * w_atom * p

cloud_co_total = np.sum(list(cloud_co_signal_by_class.values()), axis=0)
print(
    f"fraction of the original cloud surviving all clearouts: {cloud_co_survival:.4f}"
)
print("per-class absolute populations with clearouts:")
for m in sorted(cloud_co_perfect, key=lambda m: -cloud_co_perfect[m]):
    print(f"  m={m:+d}: {cloud_co_perfect[m]:.5f}")

# %%
plot_swept_readout(
    [
        (
            cloud_total,
            dict(color="0.6", lw=1.0, label="cloud, no clearouts"),
        ),
        (
            single_co_total,
            dict(
                color="tab:blue",
                lw=1.0,
                ls="--",
                label="selected atom only, with clearouts",
            ),
        ),
        (
            cloud_co_total,
            dict(
                color="tab:red",
                lw=1.4,
                label="full cloud, clearouts after pulses "
                + ", ".join(str(p) for p in clearout_windows),
            ),
        ),
    ],
    cloud_co_perfect,
    "Swept excited-state readout of the cloud with perfect clearouts in every "
    "allowed window",
)
plt.show()

# %% [markdown]
# ### What the clearouts do — two distinct things
#
# (Numbers quoted from the printed summaries above.)
#
# **1. They remove the unselected cloud, as designed.** The post-slice
# clearout (after pulse 0) kills any atom the 380 µs slice did not excite;
# ~10% of the 200 nK cloud survives to detection. The Doppler pedestal
# collapses — the m = −1 reservoir, which without clearouts held as much
# population as the m = +1 port (~0.21 of the whole cloud), drops to
# $8\times10^{-4}$, and the spectrum tightens onto a single sharp peak.
#
# **2. They *purify the interferometer itself* — at a price.** The ideal arm
# walk says the clearout windows are clean (both arms excited, nothing in
# ground), but the genuine quantum state disagrees: the imperfect square π
# pulses leave 2-9% *coherent* ground population behind at each window
# (compounding to ~20% by pulse 11 if never cleaned). Two consequences:
#
# * Even the perfectly sliced atom only survives all five windows with weight
#   ~0.77 — the clearouts post-select away 23% of the *good* atoms, because
#   that amplitude genuinely was in the ground manifold when the light fired.
# * The removed amplitude was not inert junk: the arms revisit those ground
#   rungs, so later resonant pulses coherently re-absorb the residue into the
#   recombination. Removing it changes the output dramatically — and **towards
#   the ideal interferometer**: renormalised, ~95% of the surviving atom exits
#   in m = +1, the intended single φ = 0 output port, while the m = +3 class
#   collapses from 0.28 (no clearouts, cf. the single-atom figure above) to
#   ~0.01. The prominent "m = +3 port" of the no-clearout spectra is thus
#   revealed as parasitic multi-path interference that clearouts largely
#   remove. Readout-wise: with clearouts, the φ = 0 spectrum is essentially
#   one peak at +4.7 kHz.
#
# **What remains is the instrument.** Clearouts fix the *sample*, not the
# readout: the oscillatory 10⁻³–10⁻⁴ sinc-sidelobe floor of the square imaging
# pulse is untouched and now sets the background everywhere. The next
# improvement would have to come from shaping the imaging pulse.

# %% [markdown]
# ### Model caveats
#
# * The readout is ensemble-averaged: no quantum projection noise, detection
#   noise, or shot-to-shot drift. The 461 nm flashes are assumed perfect
#   (every ground atom removed, none of the excited touched).
# * The imaging beam is spatially uniform (the recoil-class merge in
#   `lmt_sim.readout` is exact only then), and transverse Doppler shifts are
#   ignored.
# * The cloud is a deterministic 41-point velocity grid at 200 nK, ±3σ. The
#   grid spacing (0.94 kHz in Doppler units) is finer than the imaging
#   linewidth, so the pedestals are smooth, but structure much narrower than
#   the spacing would not be resolved.
# * The record itself contains no clearout events, so the clearout timings
#   are idealised: instantaneous, perfect (every ground atom removed, excited
#   amplitudes untouched), fired immediately after the qualifying pulses. Real
#   461 nm flashes have finite duration and sit somewhere inside the free-fall
#   gaps; neither changes the populations here (free fall does not mix the
#   manifolds), but photon scattering into the excited state is not modelled.
