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
# # N-pulse LMT sequence -- shot-by-shot filmstrip
#
# Same large-momentum-transfer sequence as `n_pulse_lmt_sequence.ipynb` (a $\pi/2$ beam splitter, $N$ accelerating and $N$ decelerating $\pi$ pulses on the top arm, a $\pi$ mirror, the same $N+N$ pulses on the bottom arm, a final $\pi/2$ recombiner, and a trailing free fall) but run on a thermal velocity ensemble so the synthetic camera has something to image.
#
# Each cell below renders a per-event filmstrip via `lmt_sim.imaging.plot_filmstrip`. `LMT_N` controls how many pulses per arm; the call cells sweep the mirror phase $\phi$ across $[0, 2\pi]$.

# %%
import sys
sys.path.insert(0, '..')

# %%
import numpy as np
from scipy import constants

import lmt_sim.lmt_sequence as seq
from lmt_sim.lmt_simulation import RABI_FREQ, T_PI, RECOIL_FREQUENCY_HZ
from lmt_sim.imaging import plot_filmstrip

# %matplotlib inline

# %%
# Thermal ensemble (Maxwell-Boltzmann, 1D z-velocity).
# Smaller than the MZ notebook because each LMT shot runs ~20 pulses per atom.
N_ATOMS = 50
TEMPERATURE = 200e-9
MASS_ATOM = 87 * constants.atomic_mass
sigma_v = np.sqrt(constants.k * TEMPERATURE / MASS_ATOM)
velocities = np.random.default_rng(0).normal(0, sigma_v, size=N_ATOMS)

LMT_N = 4  # pulses per arm


# %%
def build_lmt_sequence(N, phi):
    """Same sequence builder as notebooks/n_pulse_lmt_sequence.ipynb cell 5.

    Pi/2 beam splitter, N pi pulses accelerating then decelerating the top arm,
    pi mirror, N pi pulses accelerating then decelerating the bottom arm, pi/2
    recombiner with phase 4*phi, plus a trailing freefall so the clouds separate
    visibly in the final shot.
    """
    s = []

    s.append(seq.Pulse(
        k=+1, detuning_hz=+1 * RECOIL_FREQUENCY_HZ, phi=0.0,
        label='BS pi/2', rabi_frequency=RABI_FREQ, duration=T_PI / 2,
    ))

    # (k, detuning_multiplier) for the i-th pulse in an arm: (-1,-3),(+1,+5),(-1,-7),...
    kicks = [
        ((-1 if i % 2 == 0 else +1),
         (-1 if i % 2 == 0 else +1) * (2 * (i + 1) + 1))
        for i in range(N)
    ]

    for i, (k, d) in enumerate(kicks, start=1):
        s.append(seq.Pulse(k=k, detuning_hz=d * RECOIL_FREQUENCY_HZ, phi=0.0,
                           label=f'top accel {i}', rabi_frequency=RABI_FREQ, duration=T_PI))
    for i, (k, d) in enumerate(reversed(kicks), start=1):
        s.append(seq.Pulse(k=k, detuning_hz=d * RECOIL_FREQUENCY_HZ, phi=0.0,
                           label=f'top decel {i}', rabi_frequency=RABI_FREQ, duration=T_PI))

    s.append(seq.Pulse(
        k=+1, detuning_hz=+1 * RECOIL_FREQUENCY_HZ, phi=phi,
        label='mirror pi', rabi_frequency=RABI_FREQ, duration=T_PI,
    ))

    for i, (k, d) in enumerate(kicks, start=1):
        s.append(seq.Pulse(k=k, detuning_hz=d * RECOIL_FREQUENCY_HZ, phi=phi,
                           label=f'bottom accel {i}', rabi_frequency=RABI_FREQ, duration=T_PI))
    for i, (k, d) in enumerate(reversed(kicks), start=1):
        s.append(seq.Pulse(k=k, detuning_hz=d * RECOIL_FREQUENCY_HZ, phi=phi,
                           label=f'bottom decel {i}', rabi_frequency=RABI_FREQ, duration=T_PI))

    s.append(seq.Pulse(
        k=+1, detuning_hz=+1 * RECOIL_FREQUENCY_HZ, phi=4 * phi,
        label='BS pi/2 final', rabi_frequency=RABI_FREQ, duration=T_PI / 2,
    ))

    # Trailing freefall so the m-state separation is visible in the final shot.
    s.append(seq.Freefall(duration=T_PI, label='freefall'))
    return s


def plot_lmt_filmstrip(phi, N=LMT_N):
    sequence = build_lmt_sequence(N=N, phi=phi)
    return plot_filmstrip(
        sequence, velocities,
        title=f'LMT $N={N}$ filmstrip at $\\phi = {phi / np.pi:.3f}\\pi$ (each panel autoscaled)',
        desc=f'N={N}, phi={phi / np.pi:.3f}pi',
        panel_width=1.6, panel_height=3.2,
        # Lossy threshold: keeps weights to ~0.1% of exact while running ~6x faster.
        # Plenty for visualisation; bump down to 1e-9 if you need exact channel weights.
        discard_threshold=1e-6,
    )


# %%
plot_lmt_filmstrip(0.0)

# %%
plot_lmt_filmstrip(np.pi / 2)

# %%
plot_lmt_filmstrip(np.pi)

# %%
plot_lmt_filmstrip(3 * np.pi / 2)

# %%
plot_lmt_filmstrip(2 * np.pi)
