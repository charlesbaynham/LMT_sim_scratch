# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: lmt-sim-scratch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Declarative symmetric Mach-Zehnder -- RID 75603
#
# The full declarative symmetric LMT Mach-Zehnder
# (`DeclarativeLMTGlobalSymmetricMachZehnder` in icl_experiments, module
# `repository.LMT.lmt_declarative_global`, ref `27b19683`) run on 2026-06-27 at
# a 10-pulse launch and `n_recoils=2`. Submitted as a no-axes single point x5
# repeats; the schedule is deterministic so the five identical shots dedup to a
# single archived record. The dump is embedded here so the script is
# self-contained (this repo has no RDS access).
#
# Schedule shape: **30 clock pulses = 1 velocity-selection slice + 10 launch
# pulses + 19 interferometer pulses** (the symmetric builder makes the
# bs1->bs2 block `3 + 8*n_recoils` = 19 pulses for `n_recoils=2`). Unlike the
# shaped-double-launch showcase, every pulse here is a plain square pulse at
# the base switch carrier (switch offset 0 Hz); the only per-pulse structure is
# beam direction (up/down) and the long 380 us slice vs the 55/68 us
# launch/MZ pulses.

# %%
import sys

sys.path.insert(0, "..")

import numpy as np

import lmt_sim.lmt_sequence as seq

# %%
# Genuine new-format pulse dump from ARTIQ RID 75603
# (DeclarativeLMTGlobalSymmetricMachZehnder): float64 SI pulse_record_flat as
# emitted by PulseDMARecording in icl_experiments.
pulse_record_flat = np.array([
    30.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0011392000000000002, 0.0019425760000000001, 0.0020322960000000003, 0.002109016, 0.002198736,
    0.0022754560000000004, 0.002365176, 0.0024418960000000003, 0.002531616, 0.002608336, 0.0026980560000000003,
    0.002974792, 0.003064512, 0.0031412320000000003, 0.003217952, 0.003307672, 0.003447392,
    0.0035371120000000002, 0.0036268320000000004, 0.003703552, 0.0037802720000000003, 0.003869992, 0.003946712000000001,
    0.004023432, 0.004113152, 0.004252872, 0.004342592, 0.004432312, 0.0045090320000000005,
    0.004585752, 0.00038, 6.8e-05, 5.5e-05, 6.8e-05, 5.5e-05,
    6.8e-05, 5.5e-05, 6.8e-05, 5.5e-05, 6.8e-05, 5.5e-05,
    6.8e-05, 5.5e-05, 5.5e-05, 6.8e-05, 6.8e-05, 6.8e-05,
    6.8e-05, 5.5e-05, 5.5e-05, 6.8e-05, 5.5e-05, 5.5e-05,
    6.8e-05, 6.8e-05, 6.8e-05, 6.8e-05, 5.5e-05, 5.5e-05,
    6.8e-05, 79993964.06464231, 80006330.30636367, 79985391.81854664, 80022797.46822198, 79968924.6566883,
    80039264.63008031, 79952457.49482998, 80055731.79193862, 79935990.33297166, 80072198.95379695, 79919523.17111334,
    80085857.75577933, 79905864.36913097, 79925745.66179599, 80101247.71696529, 80062379.80351797, 80098026.22423503,
    80059158.31078771, 79913759.72212993, 79933641.01479495, 80074548.27197365, 79917173.85293663, 79937055.14560165,
    80089938.23315963, 80051070.31971231, 80086716.74042937, 80047848.82698204, 79925069.20593558, 79944950.4986006,
    80063238.788168, 200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0,
    200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0,
    200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0,
    200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0,
    200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0, 200000000.0,
    200000000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0,
    99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0,
    99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0,
    99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0,
    99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0, 99436000.0,
    99436000.0, 0.012, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
    2.6,
])
pulse_record_offsets = np.array([0], dtype=np.int64)

dump = seq.decode_pulse_record_flat(pulse_record_flat, pulse_record_offsets)[0]
n_pulses = len(dump.is_up)
assert n_pulses == 30, n_pulses

# %% [markdown]
# ## Sanity summary
#
# Confirm the recorded schedule matches the intended 1 slice + 10 launch + 19 MZ
# topology: a single long velocity-selection pulse, then 29 short launch/MZ
# pulses, all at switch carrier 200 MHz.

# %%
durations_us = np.round(dump.durations_s * 1e6, 1)
switch_offset = np.round(dump.switch_hz - 200e6).astype(int)

assert np.all(switch_offset == 0), np.unique(switch_offset)
n_up = int(np.sum(dump.is_up))
slice_idx = int(np.argmax(dump.durations_s))
assert slice_idx == 0 and durations_us[0] == 380.0, (slice_idx, durations_us[0])

print(f"n_pulses           = {n_pulses}")
print(f"up / down          = {n_up} / {n_pulses - n_up}")
print(f"slice pulse        = index {slice_idx}, {durations_us[0]} us, "
      f"setpoint {dump.delivery_setpoint[0]} V")
print(f"launch+MZ durations= {sorted(set(durations_us[1:].tolist()))} us")
print(f"switch carriers    = {sorted(set((dump.switch_hz).tolist()))} Hz")
print(f"delivery carriers  = {sorted(set((dump.delivery_hz).tolist()))} Hz")
