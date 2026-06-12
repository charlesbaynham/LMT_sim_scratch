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



# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from tqdm import tqdm
# from tqdm.notebook import tqdm

import sys
sys.path.insert(0, '..')

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
np.random.seed(42)

T_SPECTROSCOPY = 45e-6
T_SLICE = T_SPECTROSCOPY * 5

RABI_FREQ_SPECTROSCOPY = 1.0 / (2.0 * T_SPECTROSCOPY)

MASS_ATOM = constants.atomic_mass * 87.0
TRANSITION_WAVELENGTH = 698e-9

TEMPERATURE = 200e-9
N_ATOMS = 30

N_DURATION_POINTS = 300
MAX_AREA_PI = 6.0

# Single-photon recoil frequency
F_RECOIL = constants.h / (MASS_ATOM * TRANSITION_WAVELENGTH**2)
BASE_DETUNING_HZ = F_RECOIL / 2  # Same detuning as MZ


# %%
sigma_v = np.sqrt(constants.k * TEMPERATURE / MASS_ATOM)
velocities = np.random.normal(0.0, sigma_v, size=N_ATOMS)
pulse_durations = np.linspace(0.0, MAX_AREA_PI * T_SPECTROSCOPY, N_DURATION_POINTS)
pulse_area_over_pi = pulse_durations / T_SPECTROSCOPY

print(f"sigma_v = {sigma_v:.3e} m/s")
print(f"Max pulse duration = {pulse_durations[-1] * 1e6:.1f} us ({MAX_AREA_PI:.1f} pi area)")


# %%
def calc_sliced_pulse_excitation_borde(
    spectroscopy_pulse_duration,
    slice_pulse_duration,
    detuning_hz=0.0,
    initial_velocity_z=0.0,
):
    # For the purposes of this simualtion, we'll assume that the two pulses have
    # their power tuned for maximum excitation

    rabi_freq_slice = 1.0 / (2.0 * slice_pulse_duration)

    state = make_atom_states(
        initial_velocity_z=initial_velocity_z,
        c0=1,
        c1=0,
    )

    omega_laser = 2.0 * np.pi * (TRANSITION_FREQUENCY + detuning_hz)

    state = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=0.0,
        z=0.0,
        vz=initial_velocity_z,
        inverse=False,
    )

    # Do a velocity slice
    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=slice_pulse_duration,
        pulse_rabi_freq=rabi_freq_slice,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )

    # Clear out the ground state
    state_slice_lab = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=slice_pulse_duration,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob_slice, excited_prob_slice = calculate_ground_and_excited_probabilities(
        state_slice_lab,
    )
    atom_cleared_out = np.random.rand() < ground_prob_slice
    if atom_cleared_out:
        return float("nan")

    # Do a spectroscopy pulse
    state = pulse_interaction_in_borde_representation(
        state,
        pulse_detuning=detuning_hz,
        t_pulse=spectroscopy_pulse_duration,
        pulse_rabi_freq=RABI_FREQ_SPECTROSCOPY,
        pulse_phase=0.0,
        k_sign=+1,
        k_wavevector=K_WAVEVECTOR,
        vz=initial_velocity_z,
    )

    state_final_lab = transform_state_vector(
        state,
        omega_laser=omega_laser,
        t=slice_pulse_duration + spectroscopy_pulse_duration,
        z=0.0,
        vz=initial_velocity_z,
        inverse=True,
    )

    ground_prob, excited_prob = calculate_ground_and_excited_probabilities(
        state_final_lab,
    )

    total_prob = ground_prob + excited_prob
    return excited_prob / total_prob



# %%
excitation_traces = np.empty((N_ATOMS, N_DURATION_POINTS))

for ind_atom, velocity in enumerate(tqdm(velocities, desc="Simulating atoms")):
    for ind_duration, spectroscopy_pulse_duration in enumerate(pulse_durations):
        excitation_traces[ind_atom, ind_duration] = calc_sliced_pulse_excitation_borde(
            spectroscopy_pulse_duration=spectroscopy_pulse_duration,
            slice_pulse_duration=T_SLICE,
            detuning_hz=BASE_DETUNING_HZ,
            initial_velocity_z=velocity,
        )

# %%
mean_excitation = np.nanmean(excitation_traces, axis=0)
std_excitation = np.nanstd(excitation_traces, axis=0)

ideal_excitation = np.array([
    calc_sliced_pulse_excitation_borde(
        spectroscopy_pulse_duration=pulse_duration,
        slice_pulse_duration=T_SLICE,
        detuning_hz=BASE_DETUNING_HZ,
        initial_velocity_z=0.0,
    )
    for pulse_duration in pulse_durations
])

# %%
num_surviving_atoms = np.sum(np.isfinite(excitation_traces[:,0]))

print(f"Survival probablility: {100*num_surviving_atoms / N_ATOMS:.1f}% ({num_surviving_atoms} / {N_ATOMS})")

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for trace in excitation_traces:
    # print(trace)
    ax.plot(pulse_area_over_pi, trace, color="tab:blue", alpha=0.2, linestyle="", marker=".", markersize=2)

ax.plot(pulse_area_over_pi, ideal_excitation, color="black", lw=1.5, label="Ideal (v=0)")
ax.fill_between(
    pulse_area_over_pi,
    mean_excitation - std_excitation,
    mean_excitation + std_excitation,
    color="tab:orange",
    alpha=0.25,
    label=r"Ensemble mean $\pm 1\sigma$",
)
ax.plot(pulse_area_over_pi, mean_excitation, color="tab:orange", lw=2.0, label="Ensemble mean")

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
