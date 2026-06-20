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
# # Fringes vs detuning offset across Rabi frequency

# %%
import sys

sys.path.insert(0, "..")

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import lmt_sim.lmt_simulation as sim
import lmt_sim.lmt_sequence as seq

N_LMTs = 4
REC = sim.RECOIL_FREQUENCY_HZ


# %%
def get_sequence(N, phi, rabi, detuning_offset_hz=0.0):
    do = detuning_offset_hz
    t_pi = 0.5 / rabi
    s = []
    s.append(
        seq.Pulse(k=+1, detuning_hz=+1 * REC + do, phi=0.0, label="BS",
                  rabi_frequency=rabi, duration=t_pi / 2)
    )
    top = []
    for i in range(N):
        k = -1 if i % 2 == 0 else +1
        dm = (-1 if i % 2 == 0 else +1) * (2 * (i + 1) + 1)
        top.append((k, dm))
        s.append(
            seq.Pulse(k=k, detuning_hz=dm * REC + do, phi=0.0, label=f"ta{i}",
                      rabi_frequency=rabi, duration=t_pi)
        )
    for k, dm in reversed(top):
        s.append(
            seq.Pulse(k=k, detuning_hz=dm * REC + do, phi=0.0, label="td",
                      rabi_frequency=rabi, duration=t_pi)
        )
    s.append(
        seq.Pulse(k=+1, detuning_hz=+1 * REC + do, phi=phi, label="mirror",
                  rabi_frequency=rabi, duration=t_pi)
    )
    bot = []
    for i in range(N):
        k = -1 if i % 2 == 0 else +1
        dm = (-1 if i % 2 == 0 else +1) * (2 * (i + 1) + 1)
        bot.append((k, dm))
        s.append(
            seq.Pulse(k=k, detuning_hz=dm * REC + do, phi=phi, label=f"ba{i}",
                      rabi_frequency=rabi, duration=t_pi)
        )
    for k, dm in reversed(bot):
        s.append(
            seq.Pulse(k=k, detuning_hz=dm * REC + do, phi=phi, label="bd",
                      rabi_frequency=rabi, duration=t_pi)
        )
    s.append(
        seq.Pulse(k=+1, detuning_hz=+1 * REC + do, phi=phi * 4, label="BSf",
                  rabi_frequency=rabi, duration=t_pi / 2)
    )
    s.append(seq.Freefall(duration=t_pi, label="freefall"))
    return s


initial_state = sim.make_atom_states(
    position_x=0.0, position_y=0.0, position_z=0.0,
    initial_velocity_z=0.0, c0=1.0, c1=0.0,
)


def run_phi(phi, detuning_offset_hz, rabi):
    sequence = get_sequence(N_LMTs, phi, rabi, detuning_offset_hz)
    result = seq.run_pulse_sequence_in_lab_frame(
        initial_state, pulse_sequence=sequence, discard_threshold=1e-9
    )
    if result is None:
        raise RuntimeError("Atom was cleared out by the sequence")
    state, _, _ = result
    g, e = sim.calculate_ground_and_excited_probabilities(state)
    return e / (g + e)


# %%
phis = np.linspace(0, 2 * np.pi, 100)
offsets = np.linspace(-1.0, 1.0, 5)  # units of recoil frequency
mults = [0.01, 1.0, 100.0]
titles = {0.01: "Rabi x0.01", 1.0: "Rabi x1 (normal)", 100.0: "Rabi x100"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, mult in zip(axes, mults):
    rabi = mult * sim.RABI_FREQ
    for off in offsets:
        exc = Parallel(n_jobs=-1)(
            delayed(run_phi)(phi, off * REC, rabi)
            for phi in tqdm(phis, desc=f"{titles[mult]} off={off:+.1f}")
        )
        ax.plot(phis, exc, lw=1.6, label=f"{off:+.1f}x")
    ax.set_title(f"{titles[mult]}  ($f_R$={rabi:.0f} Hz)")
    ax.set_xlabel("Phase")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Excitation Probability")
axes[2].legend(title="detuning offset\n(x $f_{rec}$)", frameon=False, fontsize=9)
fig.suptitle(
    f"Fringes vs detuning offset across Rabi frequency, N={N_LMTs} "
    f"($f_{{rec}}$={REC:.0f} Hz)"
)
fig.tight_layout()
