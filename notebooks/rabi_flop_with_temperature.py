# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lmt-sim-scratch (3.13.5)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Temperature-Broadened Rabi Flop vs Pulse Duration
#
# This notebook simulates a single pulse with variable duration and plots the ensemble-averaged Rabi flop over many `pi` pulse areas.
#
# Velocities are sampled from a 1D Maxwell-Boltzmann distribution, matching the temperature treatment in `mach_zehnder_with_temperature.ipynb`.

# %%
import sys

import sys

sys.path.insert(0, "..")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm
# from tqdm.notebook import tqdm

import sys

sys.path.insert(0, "..")

import version_info as vs
from lmt_sim.lmt_simulation import (
    make_atom_states,
    transform_state_vector,
    pulse_interaction_in_borde_representation,
    calculate_ground_and_excited_probabilities,
    K_WAVEVECTOR,
    TRANSITION_FREQUENCY,
)

# %matplotlib inline

# %%
# np.random.seed(42)

T_PI = 45e-6
RABI_FREQ = 1.0 / (2.0 * T_PI)

MASS_ATOM = constants.atomic_mass * 87.0
TRANSITION_WAVELENGTH = 698e-9

TEMPERATURE = 200e-9
N_ATOMS = 300

N_DURATION_POINTS = 300
MAX_AREA_PI = 20.0

# Single-photon recoil frequency
F_RECOIL = constants.h / (MASS_ATOM * TRANSITION_WAVELENGTH**2)
BASE_DETUNING_HZ = F_RECOIL / 2  # Same detuning as MZ


# %%
sigma_v = np.sqrt(constants.k * TEMPERATURE / MASS_ATOM)
velocities = np.random.normal(0.0, sigma_v, size=N_ATOMS)
pulse_durations = np.linspace(0.0, MAX_AREA_PI * T_PI, N_DURATION_POINTS)
pulse_area_over_pi = pulse_durations / T_PI

print(f"sigma_v = {sigma_v:.3e} m/s")
print(
    f"Max pulse duration = {pulse_durations[-1] * 1e6:.1f} us ({MAX_AREA_PI:.1f} pi area)"
)


# %%
def calc_single_pulse_excitation_borde(
    pulse_duration,
    detuning_hz=0.0,
    initial_velocity_z=0.0,
):
    state = make_atom_states(
        initial_velocity_z=initial_velocity_z,
        c0=1,
        c1=0,
    )

    transform_detuning_hz = detuning_hz

    state = transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=pulse_duration,
        pulse_rabi_freq=RABI_FREQ,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )

    state = transform_state_vector(
        state,
        detuning_hz=transform_detuning_hz,
        t=pulse_duration,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob, excited_prob = calculate_ground_and_excited_probabilities(state)

    total_prob = ground_prob + excited_prob
    return excited_prob / total_prob


# %%
excitation_traces = np.empty((N_ATOMS, N_DURATION_POINTS))

for ind_atom, velocity in enumerate(tqdm(velocities, desc="Simulating atoms")):
    for ind_duration, pulse_duration in enumerate(pulse_durations):
        excitation_traces[ind_atom, ind_duration] = calc_single_pulse_excitation_borde(
            pulse_duration=pulse_duration,
            detuning_hz=BASE_DETUNING_HZ,
            initial_velocity_z=velocity,
        )

mean_excitation = np.mean(excitation_traces, axis=0)
std_excitation = np.std(excitation_traces, axis=0)

ideal_excitation = np.array(
    [
        calc_single_pulse_excitation_borde(
            pulse_duration=pulse_duration,
            detuning_hz=BASE_DETUNING_HZ,
            initial_velocity_z=0.0,
        )
        for pulse_duration in pulse_durations
    ]
)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for trace in excitation_traces:
    ax.plot(pulse_area_over_pi, trace, color="tab:blue", alpha=0.04)

ax.plot(
    pulse_area_over_pi, ideal_excitation, color="black", lw=1.5, label="Ideal (v=0)"
)
ax.fill_between(
    pulse_area_over_pi,
    mean_excitation - std_excitation,
    mean_excitation + std_excitation,
    color="tab:orange",
    alpha=0.25,
    label=r"Ensemble mean $\pm 1\sigma$",
)
ax.plot(
    pulse_area_over_pi,
    mean_excitation,
    color="tab:orange",
    lw=2.0,
    label="Ensemble mean",
)

ax.set_xlabel(r"Pulse area ($\pi$ units)")
ax.set_ylabel("Excitation fraction")
ax.set_xlim(0, MAX_AREA_PI)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(
    rf"Single-pulse Rabi flop with thermal dephasing, {N_ATOMS} atoms at T = {TEMPERATURE * 1e9:.0f} nK"
)
fig.tight_layout()
vs.tag_plot(small=True)

# %%
